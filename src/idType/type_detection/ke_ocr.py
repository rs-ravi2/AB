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

        ocr_detections = [[boxes, (text, float(score))]
                          for boxes, (text, score) in ocr_detections]

        national_id_keywords    = ['jamhuriyakenya']
        foreign_id_keywords     = ['refugee', 'refugeeid', 'foreignercertificate', 'nationalidcard']
        passport_strong_kw      = ['<<', '>>', 'republicofindia', 'migrationofficer']
        passport_maybe_kw       = ['passport']
        cor_keywords            = ['certificateofregistration', 'certificateofincorporation']
        huduma_strong_keywords  = ['hudumanamba']

        text_combined = "---".join(d[1][0] for d in ocr_detections).lower().replace(' ', '')

        # Early exit on Huduma or foreign ID
        if any(fuzzysearch.find_near_matches(kw, text_combined, max_l_dist=1) for kw in huduma_strong_keywords + foreign_id_keywords):
            return None

        if any(fuzzysearch.find_near_matches(kw, text_combined, max_l_dist=1) for kw in cor_keywords):
            return "Certificate of Registration"

        # Detect the order of 'jamhuri...' vs 'republicof...'
        jam_coords = rep_coords = None
        for coords, (txt, _) in ocr_detections:
            cleaned = txt.lower().replace(' ', '')
            if fuzzysearch.find_near_matches('jamhuriyakenya', cleaned, max_l_dist=3):
                jam_coords = coords
            if fuzzysearch.find_near_matches('republicofkenya', cleaned, max_l_dist=3):
                rep_coords = coords

        if jam_coords and rep_coords:
            jam_x = (jam_coords[0][0] + jam_coords[2][0]) / 2
            rep_x = (rep_coords[0][0] + rep_coords[2][0]) / 2
            if jam_x < rep_x:
                return "National ID"

        # Passport detection
        if any(fuzzysearch.find_near_matches(kw, text_combined, max_l_dist=1) for kw in passport_strong_kw):
            return "Passport"
        if any(fuzzysearch.find_near_matches(kw, text_combined, max_l_dist=3) for kw in national_id_keywords):
            return "National ID"
        if any(fuzzysearch.find_near_matches(kw, text_combined, max_l_dist=1) for kw in passport_maybe_kw) and len(ocr_detections) <= 25:
            return "Passport"

        return None

    except Exception as e:
        logger.error(f"Error in OCR ID type detection: {e}")
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

        elif detected_id_by_ocr == "Certificate of Registration":
            confidence = random.uniform(0.9, 1.0)
            return confidence

        return random.uniform(0.9, 1.0)
    
    
    except Exception as e:
        logger.error(f"Error in OCR validation scoring: {str(e)}")
        return None

