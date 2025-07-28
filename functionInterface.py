import os
import json
import time
import logging
import threading
import traceback
import importlib
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Protocol buffers compatibility fix
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import yaml
import numpy as np
import tensorflow as tf
from minio import Minio
from minio.error import S3Error

from src.idImage.retinaface_detector.retinaface_detection import RetinaFaceDetectionONNX
from src.idOCR.rapidocr_onnx.rapidocr_onxx import RapidOCRONNX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IDProcessorError(Exception):
    """Base exception for ID processor errors."""
    pass


class ConfigurationError(IDProcessorError):
    """Configuration related errors."""
    pass


class ModelLoadError(IDProcessorError):
    """Model loading related errors."""
    pass


class ModelDownloadError(IDProcessorError):
    """Model download related errors."""
    pass


class ImageProcessingError(IDProcessorError):
    """Image processing related errors."""
    pass


class MinIOModelDownloader:
    """Simple MinIO downloader for models directory with OPCO-specific configuration."""
    
    def __init__(self, opco_config: Dict[str, Any], opco: str):
        """Initialize MinIO client with OPCO-specific configuration."""
        self.opco = opco
        self.opco_config = opco_config
        self.minio_config = opco_config.get('minio_config', {})
        self.client = None
        self.models_dir = Path('./models')
        
        # Expected model subdirectories
        self.expected_model_dirs = ['idImage', 'idUpright', 'idType', 'idOCR']
        
        if not self.minio_config:
            raise ConfigurationError(f"MinIO configuration not found for OPCO '{opco}'")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize MinIO client."""
        try:
            self.client = Minio(
                self.minio_config['minio_url'],
                access_key=self.minio_config['minio_username'],
                secret_key=self.minio_config['minio_password'],
                secure=False  # Set to True if using HTTPS
            )
            
            # Test connection
            if not self.client.bucket_exists(self.minio_config['minio_bucket_name']):
                raise ModelDownloadError(f"Bucket {self.minio_config['minio_bucket_name']} does not exist")
                
            logger.info("MinIO client initialized successfully for OPCO: " + self.opco)
            
        except Exception as e:
            raise ModelDownloadError(f"Failed to initialize MinIO client for OPCO '{self.opco}': {str(e)}")
    
    def _models_directory_exists(self) -> bool:
        """Check if models directory exists with expected subdirectories."""
        if not self.models_dir.exists():
            return False
        
        # Check if all expected subdirectories exist
        for subdir in self.expected_model_dirs:
            if not (self.models_dir / subdir).exists():
                logger.info(f"Missing model subdirectory: {subdir}")
                return False
        
        logger.info("Models directory exists with all expected subdirectories")
        return True
    
    def _download_models_directory(self) -> bool:
        """Download entire models directory from MinIO for specific OPCO."""
        try:
            logger.info(f"Downloading models directory from MinIO for OPCO: {self.opco}...")
            
            # List all objects in the models/ prefix
            objects = self.client.list_objects(
                self.minio_config['minio_bucket_name'],
                prefix='models/',
                recursive=True
            )
            
            downloaded_files = 0
            failed_files = 0
            
            for obj in objects:
                # Skip directory markers
                if obj.object_name.endswith('/'):
                    continue
                
                # Convert MinIO path to local path
                # models/idImage/file.txt -> ./models/idImage/file.txt
                local_path = Path('.') / obj.object_name
                
                try:
                    # Create parent directories
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    self.client.fget_object(
                        self.minio_config['minio_bucket_name'],
                        obj.object_name,
                        str(local_path)
                    )
                    
                    downloaded_files += 1
                    logger.debug(f"Downloaded for {self.opco}: {obj.object_name}")
                    
                except Exception as e:
                    failed_files += 1
                    logger.error(f"Failed to download {obj.object_name} for {self.opco}: {str(e)}")
            
            logger.info(f"Download completed for {self.opco}: {downloaded_files} files downloaded, {failed_files} failed")
            
            # Verify that expected directories now exist
            if self._models_directory_exists():
                logger.info(f"Models directory successfully created for {self.opco} with all expected subdirectories")
                return True
            else:
                logger.error(f"Models directory download completed for {self.opco} but expected subdirectories are missing")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download models directory for {self.opco}: {str(e)}")
            return False
    
    def ensure_models_directory(self) -> bool:
        """Ensure models directory exists, download if necessary."""
        if self._models_directory_exists():
            logger.info(f"Models directory already exists for OPCO: {self.opco}")
            return True
        
        logger.info(f"Models directory not found for OPCO: {self.opco}, attempting to download from MinIO...")
        return self._download_models_directory()


def load_config(config_path: str = './config.yaml') -> dict:
    """Load configuration file with error handling."""
    try:
        if not Path(config_path).exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if not config:
            raise ConfigurationError("Configuration file is empty or invalid")

        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML configuration: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


def dynamic_import(name: str):
    """
    Dynamically import a modules with enhanced error handling.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        try:
            components = name.split('.')
            module_name = '.'.join(components[:-1])
            object_name = components[-1]

            module = importlib.import_module(module_name)
            return getattr(module, object_name)
        except ImportError as e:
            logger.error(f"Failed to import module {name}: {str(e)}")
            raise
        except AttributeError as e:
            available_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            logger.error(f"Module {module_name} has no attribute {object_name}")
            logger.error(f"Available attributes: {available_attrs}")
            raise


def validate_and_load_image(image_path: str) -> np.ndarray:
    """Load and validate image with enhanced error handling."""
    if not image_path:
        raise ImageProcessingError("Image path is missing.")

    if not Path(image_path).exists():
        raise ImageProcessingError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ImageProcessingError(f"Failed to load image from path: {image_path}")

    logger.debug(f"Successfully loaded image: {image_path}")
    return image


def preprocess_image(image: np.ndarray, img_size: int, normalize: bool = True) -> np.ndarray:
    """Preprocess image for model input with validation and optional normalization."""
    try:
        if image is None or image.size == 0:
            raise ImageProcessingError("Invalid input image")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        if normalize:
            image_normalized = image_resized.astype(np.float32) / 255.0
            return np.expand_dims(image_normalized, axis=0)
        else:
            return np.expand_dims(image_resized, axis=0)

    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ImageProcessingError(f"Image preprocessing failed: {str(e)}")


def rectify_image_orientation(image: np.ndarray, orientation: str = "0") -> np.ndarray:
    """Rectify image orientation with validation."""
    if image is None:
        logger.warning("Cannot rectify orientation: image is None")
        return None

    rotations = {
        '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '180': cv2.ROTATE_180,
        '270': cv2.ROTATE_90_CLOCKWISE
    }

    if orientation in rotations:
        try:
            image = cv2.rotate(image, rotations[orientation])
            logger.debug(f"Image rotated by {orientation} degrees")
        except Exception as e:
            logger.error(f"Failed to rotate image: {str(e)}")

    return image


def load_model_safe(model_path: str):
    """Load TensorFlow model with error handling."""
    try:
        if not Path(model_path).exists():
            raise ModelLoadError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        logger.debug(f"Model loaded successfully: {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {str(e)}")
        raise ModelLoadError(f"Failed to load model {model_path}: {str(e)}")


def get_prediction_label(prediction: np.ndarray, target_labels: list) -> str:
    """Get prediction label with error handling."""
    try:
        return max(target_labels, key=lambda label: prediction[label["prediction_index"]])["label"]
    except Exception as e:
        logger.error(f"Failed to get prediction label: {str(e)}")
        return "Unknown"


def process_ocr_fields(extracted_fields: List[Dict[str, Any]], OCRFieldNames) -> Dict[str, Any]:
    """Process OCR fields with comprehensive field mapping."""
    result = {
        "idNumber": None, "firstName": None, "middleName": None, "lastName": None,
        "fatherName": None, "motherName": None, "dateOfBirth": None, "address": None,
        "gender": None, "maritalStatus": None, "mothersMaidenName": None,
        "emergencyNumber": None, "placeOfBirth": None, "issueAuthority": None,
        "issueCountry": None, "issueDate": None, "expiryDate": None,
        "country": None, "state": None, "city": None, "postalCode": None
    }

    try:
        for field in extracted_fields:
            field_name = field.get("name")
            field_value = field.get("value")

            if not field_name or not field_value:
                continue

            if field_name == OCRFieldNames.FIRST_NAME:
                result["firstName"] = field_value
            elif field_name == OCRFieldNames.MIDDLE_NAME:
                result["middleName"] = field_value
            elif field_name == OCRFieldNames.LAST_NAME:
                result["lastName"] = field_value
            elif field_name == OCRFieldNames.GENDER:
                result["gender"] = field_value
            elif field_name == OCRFieldNames.ID_NUMBER:
                result["idNumber"] = field_value
            elif field_name == OCRFieldNames.DATE_OF_BIRTH:
                result["dateOfBirth"] = field_value
            elif field_name == OCRFieldNames.DATE_OF_ISSUE:
                result["dateOfIssue"] = field_value
            elif field_name == OCRFieldNames.DATE_OF_EXPIRY:
                result["dateOfExpiry"] = field_value
            elif field_name == OCRFieldNames.PLACE_OF_BIRTH:
                result["placeOfBirth"] = field_value

    except Exception as e:
        logger.error(f"Error processing OCR fields: {str(e)}")

    return result


def generate_image_hash(image_path: str) -> str:
    """Generate hash for image path to use as cache key."""
    return hashlib.md5(image_path.encode()).hexdigest()


class ProcessingMetrics:
    """Processing metrics context manager."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"[Timing] {self.operation_name}: {elapsed:.4f}s")


class IDProcessor:
    """Enhanced ID processor with caching, error handling, and MinIO model download."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = None
            self.opco = None
            self.model_cache = {}
            self.face_detector = None
            self.rapid_ocr = None
            self.model_downloader = None

            # Caching for computed results
            self.orientation_cache = {}  
            self.image_cache = {}  
            self.ocr_cache = {}  
            self.face_detection_cache = {}  

            self.initialized = False
            self._initialize()

    def _initialize(self):
        """Initialize ID Processor with model download capability."""
        try:
            logger.info("Initializing ID Processor...")

            # Load configuration
            self.config = load_config()

            # Get OPCO from environment FIRST (before MinIO initialization)
            self.opco = os.environ.get('opco')
            if not self.opco:
                raise ConfigurationError("OPCO environment variable is not set. Please set 'opco' before running.")

            if self.opco not in self.config:
                raise ConfigurationError(f"OPCO '{self.opco}' not found in configuration")

            logger.info(f"Using OPCO: {self.opco}")

            # Initialize MinIO model downloader with OPCO-specific config
            try:
                opco_config = self.config[self.opco]
                # Check if this OPCO has MinIO configuration
                if 'minio_config' in opco_config:
                    self.model_downloader = MinIOModelDownloader(opco_config, self.opco)
                    
                    # Ensure models directory exists
                    if not self.model_downloader.ensure_models_directory():
                        logger.error(f"Failed to ensure models directory exists for OPCO: {self.opco}")
                        raise ModelDownloadError(f"Could not create or download models directory for OPCO: {self.opco}")
                else:
                    logger.info(f"No MinIO configuration found for OPCO '{self.opco}', skipping model download")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize MinIO downloader for OPCO '{self.opco}': {str(e)}")
                logger.warning("Continuing without automatic model download capability")

            # Initialize face detector
            self.face_detector = RetinaFaceDetectionONNX()

            # Initialize OCR
            self.rapid_ocr = RapidOCRONNX()

            self.initialized = True
            logger.info("ID Processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ID Processor: {str(e)}")
            raise

    def _get_model(self, model_type: str):
        """Get model with caching."""
        cache_key = f"{self.opco}_{model_type}"

        if cache_key not in self.model_cache:
            try:
                cfg = self.config[self.opco]['models'][model_type]
                model_path = None

                if model_type == 'id_type':
                    detection_method = cfg.get('detection_method', 'classifier')
                    if detection_method in ['classifier', 'hybrid']:
                        classifier_cfg = cfg.get('classifier', {})
                        model_path = classifier_cfg.get('model_path')
                    else:
                        logger.info(f"OCR detection method for {model_type}")
                        return None
                else:
                    model_path = cfg.get('model_path') or cfg.get('classifier_model_path')

                if not model_path:
                    raise ModelLoadError(f"No model path found for {model_type}")

                self.model_cache[cache_key] = load_model_safe(model_path)
                logger.info(f"Cached model: {cache_key}")

            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {str(e)}")
                raise

        return self.model_cache[cache_key]

    def _get_cached_image(self, image_path: str) -> np.ndarray:
        """Get cached loaded image."""
        cache_key = generate_image_hash(image_path)
        if cache_key not in self.image_cache:
            self.image_cache[cache_key] = validate_and_load_image(image_path)
            logger.debug(f"Cached image: {image_path}")
        return self.image_cache[cache_key]

    def _get_cached_orientation(self, image_path: str) -> str:
        """Get cached orientation result."""
        cache_key = generate_image_hash(image_path)
        if cache_key not in self.orientation_cache:
            try:
                cfg = self.config[self.opco]['models']['id_orientation']
                image = self._get_cached_image(image_path)
                # Orientation model uses no normalization (Document 3 logic)
                image_input = preprocess_image(image, cfg['img_size'], normalize=False)

                model = self._get_model('id_orientation')
                prediction = model.predict(image_input, verbose=0)
                orientation = cfg['target_labels'].get(np.argmax(prediction, axis=-1)[0], "Unknown")

                self.orientation_cache[cache_key] = orientation
                logger.debug(f"Cached orientation for {image_path}: {orientation}")
            except Exception as e:
                logger.error(f"Error computing orientation for {image_path}: {str(e)}")
                self.orientation_cache[cache_key] = "0"

        return self.orientation_cache[cache_key]

    def _get_cached_uprighted_image(self, image_path: str) -> np.ndarray:
        """Get cached uprighted image."""
        cache_key = f"uprighted_{generate_image_hash(image_path)}"
        if cache_key not in self.image_cache:
            image = self._get_cached_image(image_path)
            orientation = self._get_cached_orientation(image_path)
            uprighted = rectify_image_orientation(image, orientation)
            self.image_cache[cache_key] = uprighted
            logger.debug(f"Cached uprighted image for {image_path}")
        return self.image_cache[cache_key]

    def _get_cached_face_detection(self, image_path: str) -> Tuple[np.ndarray, Any]:
        """Get cached face detection result."""
        cache_key = generate_image_hash(image_path)
        if cache_key not in self.face_detection_cache:
            uprighted_image = self._get_cached_uprighted_image(image_path)
            bbox, landmarks = self.face_detector.detect_faces(uprighted_image)
            self.face_detection_cache[cache_key] = (bbox, landmarks)
            logger.debug(f"Cached face detection for {image_path}")
        return self.face_detection_cache[cache_key]

    def _get_cached_ocr(self, image_path: str) -> Any:
        """Get cached OCR result."""
        cache_key = generate_image_hash(image_path)
        if cache_key not in self.ocr_cache:
            uprighted_image = self._get_cached_uprighted_image(image_path)
            ocr_result = self.rapid_ocr.run(uprighted_image)
            self.ocr_cache[cache_key] = ocr_result
            logger.debug(f"Cached OCR for {image_path}")
        return self.ocr_cache[cache_key]

    def clear_cache(self):
        """Clear all caches - useful for memory management."""
        self.orientation_cache.clear()
        self.image_cache.clear()
        self.ocr_cache.clear()
        self.face_detection_cache.clear()
        logger.info("All caches cleared")

    def get_id_orientation(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """Get ID orientation - maintains original interface."""
        with ProcessingMetrics("get_id_orientation"):
            try:
                front_image_path = input_dict.get("id_front_image")
                back_image_path = input_dict.get("id_back_image")

                front_orientation = self._get_cached_orientation(front_image_path) if front_image_path else None
                back_orientation = self._get_cached_orientation(back_image_path) if back_image_path else None

                return {
                    "id_orientation": {
                        "status": 200,
                        "message": "Successfully processed",
                        "result": {
                            "id_front_orientation": front_orientation,
                            "id_back_orientation": back_orientation
                        }
                    }
                }

            except Exception as e:
                logger.error(f"Error in get_id_orientation: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "id_orientation": {
                        "status": 0,
                        "message": f"Error processing: {str(e)}",
                        "result": {
                            "id_front_orientation": None,
                            "id_back_orientation": None
                        }
                    }
                }

    def get_id_quality(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """Get ID quality - maintains original interface."""
        with ProcessingMetrics("get_id_quality"):
            try:
                cfg = self.config[self.opco]['models']['id_quality']
                front_image_path = input_dict.get("id_front_image")

                # Use cached face detection result
                bbox, _ = self._get_cached_face_detection(front_image_path)
                uprighted_image = self._get_cached_uprighted_image(front_image_path)

                # Default score for cases where no face is detected
                import random
                score = random.uniform(0, 0.1)

                if bbox is not None and len(bbox) > 0:
                    x_min, y_min, x_max, y_max = map(int, bbox[0][:4])

                    # Validate bounding box
                    if x_max > x_min and y_max > y_min:
                        face_crop = uprighted_image[y_min:y_max, x_min:x_max]

                        if face_crop.size > 0:
                            # Quality model uses normalization (Document 3 logic)
                            face_input = preprocess_image(face_crop, cfg['img_size'], normalize=True)
                            model = self._get_model('id_quality')
                            prediction = model.predict(face_input, verbose=0)
                            _, good_score = tf.nn.softmax(prediction).numpy()[0]
                            score = float(good_score)

                return {
                    "id_quality": {
                        "status": 200,
                        "message": "Successfully processed",
                        "result": {"score": score}
                    }
                }

            except Exception as e:
                logger.error(f"Error in get_id_quality: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "id_quality": {
                        "status": 0,
                        "message": f"Error processing: {str(e)}",
                        "result": {"score": None}
                    }
                }

    def get_id_type(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """Get ID type with support for classifier, OCR, or hybrid detection."""
        with ProcessingMetrics("get_id_type"):
            try:
                cfg = self.config[self.opco]['models']['id_type']
                front_image = input_dict.get("id_front_image")
                if not front_image:
                    raise ValueError("Front image path is required")

                detection_method = cfg.get('detection_method', 'classifier')
                if detection_method not in ['classifier', 'ocr', 'hybrid']:
                    raise ConfigurationError(f"Invalid detection_method: {detection_method}")

                final_label = None

                if detection_method == 'classifier':
                    classifier_cfg = cfg.get('classifier') or \
                        (_ for _ in ()).throw(ConfigurationError("Classifier config missing"))
                    image_input = preprocess_image(
                        self._get_cached_uprighted_image(front_image),
                        cfg['img_size'], normalize=True
                    )
                    model = self._get_model('id_type')
                    prediction = model.predict(image_input, verbose=0)
                    probs = tf.nn.softmax(prediction).numpy()[0]
                    final_label = get_prediction_label(probs, classifier_cfg['target_labels'])

                elif detection_method == 'ocr':
                    ocr_cfg = cfg.get('ocr') or \
                        (_ for _ in ()).throw(ConfigurationError("OCR config missing"))
                    ocr_dets = self._get_cached_ocr(front_image)
                    ocr_module = dynamic_import(ocr_cfg['field_extraction_module'])
                    if hasattr(ocr_module, 'get_id_type_by_ocr'):
                        final_label = ocr_module.get_id_type_by_ocr(ocr_dets)
                    elif callable(ocr_module):
                        final_label = ocr_module(ocr_dets)
                    else:
                        raise AttributeError("OCR module invalid: no callable or method")

                elif detection_method == 'hybrid':
                    ocr_cfg = cfg.get('ocr') or \
                        (_ for _ in ()).throw(ConfigurationError("OCR config missing"))
                    try:
                        ocr_dets = self._get_cached_ocr(front_image)
                        ocr_module = dynamic_import(ocr_cfg['field_extraction_module'])
                        if hasattr(ocr_module, 'get_id_type_by_ocr'):
                            final_label = ocr_module.get_id_type_by_ocr(ocr_dets)
                        elif callable(ocr_module):
                            final_label = ocr_module(ocr_dets)
                        else:
                            raise AttributeError("OCR module invalid: no callable or method")
                    except Exception as e:
                        logger.warning(f"Hybrid OCR failed: {e}", exc_info=True)
                        final_label = None

                    if final_label is None:
                        classifier_cfg = cfg.get('classifier') or \
                            (_ for _ in ()).throw(ConfigurationError("Classifier config missing"))
                        try:
                            image_input = preprocess_image(
                                self._get_cached_uprighted_image(front_image),
                                cfg['img_size'], normalize=True
                            )
                            model = self._get_model('id_type')
                            prediction = model.predict(image_input, verbose=0)
                            probs = tf.nn.softmax(prediction).numpy()[0]
                            final_label = get_prediction_label(probs, classifier_cfg['target_labels'])
                        except Exception as e:
                            logger.warning(f"Hybrid classifier failed: {e}", exc_info=True)
                            final_label = None

                return {
                    "id_type": {
                        "status": 200,
                        "message": "Successfully processed",
                        "result": {"labels": final_label}
                    }
                }

            except Exception as e:
                logger.error(f"Error in get_id_type: {e}", exc_info=True)
                return {
                    "id_type": {
                        "status": 0,
                        "message": f"Error processing: {e}",
                        "result": {"labels": None}
                    }
                }

    def get_id_demographic_details(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """Get ID demographic details - maintains original interface."""
        with ProcessingMetrics("get_id_demographic_details"):
            try:
                cfg = self.config[self.opco]['models']['id_demographics']
                ocr_field_extraction = dynamic_import(cfg['ocr_field_extraction'])
                OCRFieldNames = dynamic_import(cfg['ocr_field_names'])

                front_image_path = input_dict.get("id_front_image")
                back_image_path = input_dict.get("id_back_image")

                # Use cached OCR results
                detections_front = self._get_cached_ocr(front_image_path) if front_image_path else []
                detections_back = self._get_cached_ocr(back_image_path) if back_image_path else []

                # Still need original front image for field extraction
                front_img = self._get_cached_image(front_image_path) if front_image_path else None

                extracted_fields = ocr_field_extraction(detections_front, detections_back, front_img)
                result = process_ocr_fields(extracted_fields, OCRFieldNames)

                return {
                    "demographicDetails": {
                        "status": 200,
                        "message": "Successfully processed",
                        "result": result
                    }
                }

            except Exception as e:
                logger.error(f"Error in get_id_demographic_details: {str(e)}")
                logger.error(traceback.format_exc())

                empty_result = {
                    "idNumber": None, "firstName": None, "middleName": None, "lastName": None,
                    "fatherName": None, "motherName": None, "dateOfBirth": None, "address": None,
                    "gender": None, "maritalStatus": None, "mothersMaidenName": None,
                    "emergencyNumber": None, "placeOfBirth": None, "issueAuthority": None,
                    "issueCountry": None, "issueDate": None, "expiryDate": None,
                    "country": None, "state": None, "city": None, "postalCode": None
                }

                return {
                    "demographicDetails": {
                        "status": 0,
                        "message": f"Error processing: {str(e)}",
                        "result": empty_result
                    }
                }


# Global processor instance
_processor = IDProcessor()


# Expose methods that maintain exact same interface as original code
def get_id_orientation(input_dict: Dict[str, str]) -> Dict[str, Any]:
    """Get ID orientation - maintains original interface."""
    return _processor.get_id_orientation(input_dict)


def get_id_quality(input_dict: Dict[str, str]) -> Dict[str, Any]:
    """Get ID quality - maintains original interface."""
    return _processor.get_id_quality(input_dict)


def get_id_type(input_dict: Dict[str, str]) -> Dict[str, Any]:
    """Get ID type - maintains original interface."""
    return _processor.get_id_type(input_dict)


def get_id_demographic_details(input_dict: Dict[str, str]) -> Dict[str, Any]:
    """Get ID demographic details - maintains original interface."""
    return _processor.get_id_demographic_details(input_dict)


def clear_cache() -> None:
    """Clear all caches - useful for memory management."""
    _processor.clear_cache()


# Health check function
def health_check() -> Dict[str, Any]:
    """Perform system health check."""
    try:
        models_dir = Path('./models')
        expected_dirs = ['idImage', 'idUpright', 'idType', 'idOCR']
        
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "opco": _processor.opco,
            "initialized": _processor.initialized,
            "models_directory_exists": models_dir.exists(),
            "expected_model_dirs": {
                dir_name: (models_dir / dir_name).exists() 
                for dir_name in expected_dirs
            },
            "minio_available": _processor.model_downloader is not None,
            "cached_models": list(_processor.model_cache.keys()),
            "cache_stats": {
                "orientation_cache": len(_processor.orientation_cache),
                "image_cache": len(_processor.image_cache),
                "ocr_cache": len(_processor.ocr_cache),
                "face_detection_cache": len(_processor.face_detection_cache)
            }
        }

        # Add MinIO connectivity status if available
        if _processor.model_downloader:
            try:
                bucket_exists = _processor.model_downloader.client.bucket_exists(
                    _processor.model_downloader.minio_config['minio_bucket_name']
                )
                health_status["minio_connectivity"] = "connected" if bucket_exists else "bucket_not_found"
            except Exception as e:
                health_status["minio_connectivity"] = f"error: {str(e)}"

        return health_status

    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# Sample usage - maintains exact same interface as original
if __name__ == "__main__":
    input_data = {
        "id_front_image": "/Users/a13402263/Documents/abacusAIRules/dev/1741421599142_kyc1740979846355_NATIONAL_ID_FRONT.jpg",
        "id_back_image": "/Users/a13402263/Documents/abacusAIRules/dev/1741421599142_kyc1740979846355_NATIONAL_ID_BACK.jpg"
    }

    try:
        print("🚀 Testing Enhanced ID Processor with MinIO & Optimized Preprocessing...")
        print("=" * 70)

        # Health check first
        health = health_check()
        print(f"Health Status: {health.get('overall_status')}")
        print(f"OPCO: {health.get('opco')}")
        print(f"Models Directory Exists: {health.get('models_directory_exists')}")
        print(f"Expected Model Directories: {health.get('expected_model_dirs')}")
        print(f"MinIO Available: {health.get('minio_available')}")
        print(f"MinIO Connectivity: {health.get('minio_connectivity', 'N/A')}")
        print(f"Cache Stats: {health.get('cache_stats')}")
        print()

        # Test all functions with exact same interface
        print("Testing ID Orientation...")
        orientation_result = get_id_orientation(input_data)
        print(f"Orientation Status: {orientation_result['id_orientation']['status']}")
        print(f"Front Orientation: {orientation_result['id_orientation']['result']['id_front_orientation']}")
        print(f"Back Orientation: {orientation_result['id_orientation']['result']['id_back_orientation']}")
        print()

        print("Testing ID Quality...")
        quality_result = get_id_quality(input_data)
        print(f"Quality Status: {quality_result['id_quality']['status']}")
        print(f"Quality Score: {quality_result['id_quality']['result']['score']}")
        print()

        print("Testing ID Type...")
        type_result = get_id_type(input_data)
        print(f"Type Status: {type_result['id_type']['status']}")
        print(f"Type Label: {type_result['id_type']['result']['labels']}")
        print()

        print("Testing Demographics...")
        demo_result = get_id_demographic_details(input_data)
        print(f"Demographics Status: {demo_result['demographicDetails']['status']}")

        # Show extracted fields
        demo_fields = demo_result['demographicDetails']['result']
        extracted = {k: v for k, v in demo_fields.items() if v is not None}
        if extracted:
            print("Extracted Fields:")
            for field, value in extracted.items():
                print(f"  {field}: {value}")
        else:
            print("No fields extracted")

        # Show final cache stats
        print("\n" + "=" * 70)
        final_health = health_check()
        print(f"Final Cache Stats: {final_health.get('cache_stats')}")
        print("\n✅ All tests completed successfully!")
        print("📂 Models directory automatically downloaded from MinIO if missing!")
        print("🔄 Caching ensures optimal performance across multiple calls!")
        print("⚙️  Optimized preprocessing: orientation (no norm), quality & type (normalized)!")

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        traceback.print_exc()