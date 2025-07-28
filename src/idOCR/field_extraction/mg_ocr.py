import re
import numpy as np
import cv2
import fuzzysearch

class OCRFieldNames:
    ID_NUMBER = "ID Number"
    FIRST_NAME = "First Name"
    MIDDLE_NAME = "Middle Name"
    LAST_NAME = "Last Name"
    GENDER = "Gender"
    DATE_OF_BIRTH = "Date of Birth"
    DATE_OF_ISSUE = "Date of Issue"
    DATE_OF_EXPIRY = "Date of Expiry"
    PLACE_OF_BIRTH = "Place of Birth"
    ID_TYPE = "ID Type"

def extract_idnumber(front_ocr_results, image):
    if not front_ocr_results or image is None:
        return None

    img_width, img_height = image.shape[:2]

    results = []
    for coords, text in front_ocr_results:
        if not text or not coords:
            continue
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
        cent_x = (x1 + x2 + x3 + x4) / 4
        cent_y = (y1 + y2 + y3 + y4) / 4
        results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })

    results = sorted(results, key=lambda x: x['cent_y'])

    laharana_pattern = re.compile(r'(lah[a-z]{2,6}|aharan[a-z]{0,4}|laha?rana[a-z/]*)', re.IGNORECASE)
    laharana_box = None
    for result in results:
        text = result['text'][0].strip()
        if laharana_pattern.match(text):
            laharana_box = result
            break

    def is_valid_id_candidate(text):
        cleaned = text.replace('.', '').replace(' ', '').strip()
        if not cleaned or cleaned.isalpha():
            return False
        pattern = r'^[A-Za-z0-9?/:\-]+$'
        return bool(re.fullmatch(pattern, cleaned) and any(c.isdigit() for c in cleaned))

    id_number = None
    id_number_coords = None
    id_number_score = None

    if laharana_box:
        y_ref = laharana_box['cent_y']
        x_ref = laharana_box['cent_x']
        x_margin = img_width * 0.35

        min_distance = float('inf')
        for result in results:
            dy = result['cent_y'] - y_ref
            dx = abs(result['cent_x'] - x_ref)
            if dy > 0 and dx <= x_margin:
                text = result['text'][0].replace('.', '').strip()
                if is_valid_id_candidate(text) and dy < min_distance:
                    min_distance = dy
                    id_number = text
                    id_number_coords = result['coords']
                    id_number_score = result['text'][1]

    # Fallback if short or missing ID (length exactly 3 or 6)
    if not id_number or len(id_number) in [3, 6]:
        segmented_result = extract_segmented_idnumber(front_ocr_results, image)
        if segmented_result:
            id_number = segmented_result['value']
            id_number_coords = segmented_result['coordinates']
            id_number_score = segmented_result['score']

    # Final cleanup: normalize and truncate
    if id_number:
        id_number = id_number.replace(" ", "")
        replace_map = str.maketrans({
            'I': '1', 'i': '1', 'A': '1', 'a': '1',
            'H': '4', 'h': '4', 'O': '0', 'o': '0',
            'S': '3', 's': '3', '/': '1', 'F': '7', 'f': '7',
            'g': '9', 'G': '9', 'q': '9', 'Q': '9',
            'Z': '2', 'z': '2'
        })
        id_number = id_number.translate(replace_map)
        if len(id_number) > 12:
            id_number = id_number[:12]

    return {
        'name': OCRFieldNames.ID_NUMBER,
        'value': id_number,
        'coordinates': id_number_coords,
        'score': id_number_score
    }

def extract_segmented_idnumber(front_ocr_results, image):
    if not front_ocr_results or image is None:
        return {
        'value': None,
        'coordinates': None,
        'score': None,
        'source': None
    }

    img_width, img_height = image.shape[:2]

    results = []
    for coords, text in front_ocr_results:
        if not text or not coords:
            continue
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
        cent_x = (x1 + x2 + x3 + x4) / 4
        cent_y = (y1 + y2 + y3 + y4) / 4
        results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })

    if not results:
        return None

    laharana_pattern = re.compile(r'l[a4h]*a*r*a*n[a-z/]*', re.IGNORECASE)
    laharana_box = None
    for result in results:
        box_text = result['text'][0].lower().replace(' ', '')
        if laharana_pattern.search(box_text):
            laharana_box = result
            break

    if not laharana_box:
        return None

    y_ref = laharana_box['cent_y']
    x_min_ref = min(pt[0] for pt in laharana_box['coords'])

    segment_candidates = []
    ocr_fix_map = str.maketrans({
        'I': '1', 'A': '1', 'H': '4', 'O': '0', 'S': '3', '/': '1',
        'F': '7', 'g': '9', 'q': '9', 'Z': '2', '?': '', ':': '', '.': ''
    })

    y_max_margin = img_height * 0.3
    x_max_margin = img_width * 0.6

    for result in results:
        if not (y_ref < result['cent_y'] < y_ref + y_max_margin):
            continue
        if not (x_min_ref < result['cent_x'] < x_min_ref + x_max_margin):
            continue
        raw_text = result['text'][0].strip().replace(' ', '')
        fixed_text = raw_text.translate(ocr_fix_map)

        if 2 <= len(fixed_text) <= 6 and any(c.isdigit() for c in fixed_text):
            result['corrected_text'] = fixed_text
            segment_candidates.append((result['cent_y'], result))

    if not segment_candidates:
        return None

    grouped_rows = {}
    y_tolerance = img_height * 0.02
    for y, result in segment_candidates:
        found = False
        for key in grouped_rows:
            if abs(key - y) <= y_tolerance:
                grouped_rows[key].append(result)
                found = True
                break
        if not found:
            grouped_rows[y] = [result]

    best_group = max(grouped_rows.values(), key=len)
    best_group = sorted(best_group, key=lambda b: b['cent_x'])

    if not best_group:
        return None

    id_parts = [b.get('corrected_text', b['text'][0].replace(' ', '').strip()) for b in best_group]
    joined_id = ''.join(id_parts)[:12]

    return {
        'value': joined_id,
        'coordinates': best_group[0]['coords'],
        'score': best_group[0]['text'][1],
        'source': 'segmented_id_grouping'
    }

def extract_kyc_fields(front_ocr_results, back_ocr_results, image):
    """
    Wrapper for compatibility with zm_ocr's extract_kyc_fields.
    Only extracts ID number using extract_idnumber.
    """
    id_number_dict = extract_idnumber(front_ocr_results, image)
    return [id_number_dict]
