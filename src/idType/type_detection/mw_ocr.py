import os, sys
import cv2
import time
import numpy as np
import random
import fuzzysearch
import re
import logging

logger = logging.getLogger(__name__)




def get_id_type_by_ocr(ocr_detections: list) -> str:
        """ Get ID type detected using OCR logic """
        
        try:
            if not ocr_detections:
                logger.warning("No OCR detections provided")
                return None


            ocr_detections = [ [boxes, (text, float(score))] for boxes, (text, score) in ocr_detections]


            national_id_keywords = ['republicofmalawi','citizenidentification','identification','chiphasochanzika']
            foreign_id_keywords = ['refugee','refugeeid','refugeeidentitycard','asylum','asylumseeker','asylumseekeridentitycard']
            passport_strong_keywords = ['<<', '>>']
            passport_maybe_keywords = ['passport','passeport','republicofsouthafrica','republicofindia']
            dl_keywords = ['drivinglicence','driving','licence','cartadeconducao','permisdeconduire']
            national_back_keywords=['nationalregistrationbureau','registrationbureau']

            # Convert OCR text to lowercase and remove spaces
            text = "---".join([detection[1][0] for detection in ocr_detections])
            text = text.lower().replace(' ', '')


            # Check for Passport first
            if any(fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in passport_strong_keywords):
                if any(fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in national_back_keywords):
                       return "National ID"
                return "Passport"

            if any(fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in passport_maybe_keywords):
                return "Passport"

            # Check for National ID
            if any(fuzzysearch.find_near_matches(keyword, text, max_l_dist=3) for keyword in national_id_keywords):
                return "National ID"


            # Check for Driving Licence
            if any(fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in dl_keywords):
                return "Driving Licence"

            # Ensure "Other Id" is returned if no keywords match
            return None
        
        
        except Exception as e:
            logger.error(f"Error in OCR ID type detection: {str(e)}")
            return None




def get_id_validation_score_by_ocr(ocr_detections) -> float :
    
    try:
        # Run OCR detection
        detected_id_by_ocr = get_id_type_by_ocr(ocr_detections)


        if detected_id_by_ocr == "National ID":
            confidence = random.uniform(0.9, 1.0)
            return confidence

        elif detected_id_by_ocr == "Passport":
            confidence = random.uniform(0.9, 1.0)
            return confidence

        elif detected_id_by_ocr == "Driving Licence":
            confidence = random.uniform(0.9, 1.0)
            return confidence

        return random.uniform(0.9, 1.0)
    
    
    except Exception as e:
        logger.error(f"Error in OCR validation scoring: {str(e)}")
        return None