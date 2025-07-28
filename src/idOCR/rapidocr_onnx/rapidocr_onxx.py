import os
import numpy as np
import pathlib
import tempfile
import yaml
from rapidocr_onnxruntime import RapidOCR


class RapidOCRONNX:

    def __init__(self) -> None:
        self.load()

    def load(self):
        # Get the original config path
        original_config_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yml")
        
        # Read the original config
        with open(original_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the current working directory (where models folder is located)
        current_dir = os.getcwd()
        
        # Update model paths to absolute paths
        config['Det']['model_path'] = os.path.join(current_dir, config['Det']['model_path'])
        config['Cls']['model_path'] = os.path.join(current_dir, config['Cls']['model_path'])
        config['Rec']['model_path'] = os.path.join(current_dir, config['Rec']['model_path'])
        
        # Create temporary config file
        self.temp_config_fd, self.temp_config_path = tempfile.mkstemp(suffix='.yml', text=True)
        
        try:
            # Write the modified config to temp file
            with os.fdopen(self.temp_config_fd, 'w') as temp_file:
                yaml.dump(config, temp_file, default_flow_style=False)
            
            # Initialize RapidOCR with the temporary config
            self.rapid_ocr = RapidOCR(self.temp_config_path)
            
        except Exception as e:
            # Clean up temp file if initialization fails
            self._cleanup_temp_config()
            raise e

    def _cleanup_temp_config(self):
        """Clean up temporary config file"""
        if hasattr(self, 'temp_config_path') and os.path.exists(self.temp_config_path):
            try:
                os.unlink(self.temp_config_path)
            except:
                pass  # Ignore cleanup errors

    def run(self, image: np.ndarray) -> str:
        """ Get id type detected using OCR logic """
        ocr_detections, _ = self.rapid_ocr(image)
        ocr_detections = [
            [boxes, (text, float(score))] for boxes, text, score in ocr_detections
        ]
        return ocr_detections

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_temp_config()