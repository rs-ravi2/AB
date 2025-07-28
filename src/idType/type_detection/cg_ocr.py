import os, sys
import cv2
import time
import numpy as np
import random
import fuzzysearch
import re
import logging

logger = logging.getLogger(__name__)


class ModelsConfig:
    NATIONAL_ID = ["National ID", "CARTE NATIONALE D IDENTITE"]
    PASSPORT = "Passport"
    DRIVING_LICENSE = "Driving Licence"
    CONSULAR_CARD = "consular card"


# Label mapping for standardized output
OCR_LABEL_MAPPING = {
    'national': 'National ID',
    'passport': 'Passport',
    'dl': 'Driving License',
    'merchant': 'Merchant ID',
    'student': 'Student ID',
    'non_id': None,
    'other': None
}


def get_id_type_by_ocr(ocr_detections: list) -> str:
    try:
        if not ocr_detections:
            logger.warning("No OCR detections provided")
            return None

        # Extract and preprocess text
        text = "---".join([detection[1][0] for detection in ocr_detections])
        text = text.lower().replace(' ', '').replace("'", "")

        # Keyword definitions
        national_id_keywords = ['cartenationaledidentite', 'cartenationale', 'nationaledidentite']
        passport_keywords = ['<<<<', '>>>>', '<<<', 'kkkk', 'passeport', 'passport', 'reisepass', 'chinese']
        merchant_id_keywords = ['numerodidentificationunique', 'ndocument', 'raisonsociale', 'nomcommercial']
        dl_keywords = ['conduire', 'permisdeconduire', 'categoriesdevehicules', 'pourlesquelslepermisestvalable',
                       'généraldestransportsterrestres']
        student_id_keywords = ['classe', 'lenseignement', 'scolaire', 'education', 'universite',
                               'cartedidentitescolaire',
                               'cartedetudiant', 'candidat', 'grade', 'faculte', 'ecole', 'institut',
                               'cartescolaire', 'examens']

        # Fuzzy matching for keywords
        national_id_matches = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=1)
                               for keyword in national_id_keywords]
        passport_matches = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=1)
                            for keyword in passport_keywords]
        merchant_id_matches = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=3)
                               for keyword in merchant_id_keywords]
        dl_matches = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=1)
                      for keyword in dl_keywords]
        student_id_matches = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=1)
                              for keyword in student_id_keywords]

        # Priority-based detection logic
        if any(national_id_matches):
            raw_result = "national"
        elif any(merchant_id_matches):
            # Check for override keywords within merchant matches
            if any(fuzzysearch.find_near_matches('passeport', text, max_l_dist=1)):
                raw_result = "passport"
            elif any(fuzzysearch.find_near_matches('permisdeconduire', text, max_l_dist=1)):
                raw_result = "dl"
            elif any([fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in
                      ['cartedelecteur', 'cartedidetiteconsulaire', 'cartedidentitepourrefugie']]):
                raw_result = "non_id"
            else:
                raw_result = "merchant"
        elif any(student_id_matches):
            # Check for override keywords within student matches
            if any(fuzzysearch.find_near_matches('passeport', text, max_l_dist=1)):
                raw_result = "passport"
            elif any(fuzzysearch.find_near_matches('permisdeconduire', text, max_l_dist=1)):
                raw_result = "dl"
            elif any([fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in
                      ['cartedelecteur', 'cartedidetiteconsulaire', 'cartedidentitepourrefugie']]):
                raw_result = "non_id"
            else:
                raw_result = "student"
        elif any(dl_matches):
            raw_result = "dl"
        elif any(passport_matches):
            raw_result = "passport"
        else:
            raw_result = "non_id"

        # Map to standardized labels
        standardized_label = OCR_LABEL_MAPPING.get(raw_result)

        logger.debug(f"OCR detection: raw='{raw_result}' -> standardized='{standardized_label}'")
        return standardized_label

    except Exception as e:
        logger.error(f"Error in OCR ID type detection: {str(e)}")
        return None


def get_id_validation_score_by_ocr(ocr_extractions: list, request_id_type: str = "National ID") -> float:
    """
    Get ID validation score using OCR extractions.

    Args:
        ocr_extractions: List of OCR detection results
        request_id_type: Requested ID type for validation

    Returns:
        Validation score between 0 and 1, or None if not applicable
    """
    try:
        detected_id_type = get_id_type_by_ocr(ocr_extractions)

        # Convert to internal format for scoring
        detected_internal = None
        for key, value in OCR_LABEL_MAPPING.items():
            if value == detected_id_type:
                detected_internal = key
                break

        if not detected_internal:
            return random.uniform(0.1, 0.3)

        # Scoring logic based on match with requested type
        if request_id_type in ModelsConfig.NATIONAL_ID:
            if detected_internal == 'national':
                return random.uniform(0.9, 1.0)
            elif detected_internal in ['passport', 'dl']:
                return random.uniform(0.9, 1.0)  # High confidence for valid IDs
            elif detected_internal in ['merchant', 'student']:
                return random.uniform(0.1, 0.3)  # Low confidence for non-standard IDs
            else:
                return random.uniform(0.1, 0.3)
        else:
            return None

    except Exception as e:
        logger.error(f"Error in OCR validation scoring: {str(e)}")
        return None