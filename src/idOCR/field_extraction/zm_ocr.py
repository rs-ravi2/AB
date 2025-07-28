import fuzzysearch
import re
import datetime


class OCRFieldNames:

    FIRST_NAME = "First Name"
    MIDDLE_NAME = "Middle Name"
    LAST_NAME = "Last Name"
    GENDER = "Gender"
    ID_NUMBER = "ID Number"
    DATE_OF_BIRTH = "Date of Birth"
    DATE_OF_ISSUE = "Date of Issue"
    DATE_OF_EXPIRY = "Date of Expiry"
    PLACE_OF_BIRTH = "Place of Birth"
    ID_TYPE = "ID Type"


def extract_kyc_fields(front_ocr_results, back_ocr_results, image):

    img_width, img_height = image.shape[:2]
    date_of_birth, date_of_birth_coords, date_of_birth_score, dob_cent_y = None, None, None, None
    gender_field, gender_field_score, gender_field_coords, gender_cent_y = None, None, None, None
    first_name, first_name_score, first_name_coords = None, None, None
    middle_name, middle_name_score, middle_name_coords = None, None, None
    last_name, last_name_score, last_name_coords = None, None, None

    ### Sort OCR results using centroid-y of detected polygons
    results = []
    for coords, text in back_ocr_results:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
        cent_y = (y1 + y2 + y3 + y4) / 4
        cent_x = (x1 + x2 + x3 + x4) / 4
        results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })
    sorted_results = sorted(results, key=lambda x: x['cent_y'])

    ### Find the position of Registration Card field (Being the most prominent field)
    reg_card_pos = None
    national_pos = None
    for idx, result in enumerate(sorted_results):
        text = result['text'][0].lower().replace(' ', '')
        reg_card_matches = fuzzysearch.find_near_matches('registrationcard', text, max_l_dist=1)
        national_matches = fuzzysearch.find_near_matches('national', text, max_l_dist=0)
        if national_matches:
            national_pos = idx
        if reg_card_matches:
            reg_card_pos = idx
            break

    if reg_card_pos is None and national_pos is not None:
        reg_card_pos = national_pos
            
    if not reg_card_pos:
        pass
    else:
        
        ### Iterate over next 10 fields in order from sorted results
        ### All our required fields would be present within these 10 fields
        name_fields = []
        is_name_found = False
        for result in sorted_results[reg_card_pos+1: reg_card_pos+12]:
            # Process the text to convert it to lower case and remove all whitespaces
            text = result['text'][0].lower().replace(' ', '')
            full_name_label = fuzzysearch.find_near_matches('fullname', text, max_l_dist=3)
            if full_name_label:
                continue
            
            # Finding the keyword matches which would stop the search for full name field
            end_matches_1 = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=2)
                            for keyword in ["dateofbirth", "placeofbirth", "village", "district", "placeof", "ofbirth"]]
            end_matches_2 = [fuzzysearch.find_near_matches(keyword, text, max_l_dist=0)
                            for keyword in ["sex", "xex"]]
            
            if any(end_matches_1) or any(end_matches_2):
                is_name_found = True
            
            # Check for date of birth field
            date_pattern = r'^(0?[1-9]|[12][0-9]|3[01])[-/\.]?(0?[1-9]|1[012])[-/\.]?(19[0-9]{2}|20[0-9]{2}|[0-9]{2})$'
            date_matches = re.search(date_pattern, re.sub('[^0-9]', '', text))
            if date_matches and result['cent_x'] < 0.5 * img_width:
                day = date_matches.group(1)
                month = date_matches.group(2)
                year = date_matches.group(3)
                if len(year) == 2:
                    current_year = int(str(datetime.datetime.now().year)[-2:])
                    year = "19" + year if int(year) > current_year else "20" + year
                date_of_birth = "{}-{:02d}-{:02d}".format(year, int(month), int(day))
                date_of_birth_score = result['text'][1]
                date_of_birth_coords = result['coords']
                dob_cent_y = result['cent_y']
                is_name_found = True
            # Check if the text is a name, and append to name field positions
            # If we've already found 3 name positions (first, middle, last), set name found to be True
            name_pattern = r'^[A-Za-z]{3,}(?: [A-Za-z]+)*$'
            name_matches = re.search(name_pattern, text)
            if name_matches and not is_name_found:
                name_fields.append(result)
                if len(name_fields) == 3:
                    is_name_found = True
            
            # Check for gender field
            male_gender_keywords = ['m', 'male', 'w']
            female_gender_keywords = ['f', 'fem', 'female', 't', 'temale']
            
            if re.sub('[^a-zA-Z]', '', text) in male_gender_keywords and gender_field is None:
                gender_field = 'Male'
                gender_field_score = result['text'][1]
                gender_field_coords = result['coords']
                gender_cent_y = result['cent_y']
            elif re.sub('[^a-zA-Z]', '', text) in female_gender_keywords and gender_field is None:
                gender_field = 'Female'
                gender_field_score = result['text'][1]
                gender_field_coords = result['coords']
                gender_cent_y = result['cent_y']
        
        # Extract the first, middle and last name from the name fields
        name_fields = sorted(name_fields, key=lambda x: x['cent_x'])

        # Filter incorrect name fields based on date of birth and gender field
        choose_name_fields = []
        for name_field in name_fields:
            drop_field = False
            if gender_cent_y:
                if gender_cent_y - name_field['cent_y'] <= 5:
                    drop_field = True
            if dob_cent_y:
                if dob_cent_y - name_field['cent_y'] <= 5:
                    drop_field = True
            if not drop_field:
                choose_name_fields.append(name_field)
        name_fields = choose_name_fields
        
        if len(name_fields) == 3:
            first_name = name_fields[0]['text'][0]
            first_name_score = name_fields[0]['text'][1]
            first_name_coords = name_fields[0]['coords']
            middle_name = name_fields[1]['text'][0]
            middle_name_score= name_fields[1]['text'][1]
            middle_name_coords = name_fields[1]['coords']
            last_name = name_fields[2]['text'][0]
            last_name_score = name_fields[2]['text'][1]
            last_name_coords = name_fields[2]['coords']
            
        elif len(name_fields) == 2:
            names_1 = name_fields[0]['text'][0].split(' ')
            names_2 = name_fields[1]['text'][0].split(' ')
            
            if len(names_1) >= 2:
                first_name, middle_name = names_1[:2]
                last_name = names_2[0]
            elif len(names_2) >= 2:
                first_name = names_1[0]
                middle_name, last_name = names_2[:2]
            else:
                ((x1_1, y1_1), (x1_2, y1_2)), cent1_y = name_fields[0]['coords'][:2], name_fields[0]['cent_y']
                ((x2_1, y2_1), (x2_2, y2_2)), cent2_y = name_fields[1]['coords'][:2], name_fields[1]['cent_y']
                if x2_1 < name_fields[0]['cent_x'] < x2_2 and x1_1 < name_fields[1]['cent_x'] < x1_2:
                    if cent1_y < cent2_y:
                        first_name = name_fields[0]['text'][0]
                        first_name_score = name_fields[0]['text'][1]
                        first_name_coords = name_fields[0]['coords']
                        last_name = name_fields[1]['text'][0]
                        last_name_score = name_fields[1]['text'][1]
                        last_name_coords = name_fields[1]['coords']
                    else:
                        first_name = name_fields[1]['text'][0]
                        first_name_score = name_fields[1]['text'][1]
                        first_name_coords = name_fields[1]['coords']
                        last_name = name_fields[0]['text'][0]
                        last_name_score = name_fields[0]['text'][1]
                        last_name_coords = name_fields[0]['coords']
                else:
                    first_name = names_1[0]
                    first_name_score = name_fields[0]['text'][1]
                    first_name_coords = name_fields[0]['coords']
                    last_name = names_2[0]
                    last_name_score = name_fields[1]['text'][1]
                    last_name_coords = name_fields[1]['coords']
                
                middle_name = None
                middle_name_score = None
                middle_name_coords = None
                
        elif len(name_fields) == 1:
            names = name_fields[0]['text'][0].split(' ')
            if len(names) >= 3:
                first_name, middle_name, last_name = names[:3]
                first_name_score = middle_name_score = last_name_score = name_fields[0]['text'][1]
                first_name_coords = middle_name_coords = last_name_coords = name_fields[0]['coords']
            elif len(names) == 2:
                first_name, last_name = names
                first_name_score = last_name_score = name_fields[0]['text'][1]
                first_name_coords = last_name_coords = name_fields[0]['coords']
                middle_name = None
                middle_name_score = middle_name_coords = None
            else:
                first_name = names[0]
                first_name_score = name_fields[0]['text'][1]
                first_name_coords = name_fields[0]['coords']
                middle_name, last_name = None, None
                middle_name_score = middle_name_coords = None
                last_name_score = last_name_coords = None
        else:
            first_name, middle_name, last_name = None, None, None
            first_name_score = first_name_coords = None
            middle_name_score = middle_name_coords = None
            last_name_score = last_name_coords = None

    dob_dict = {
        "name": OCRFieldNames.DATE_OF_BIRTH,
        "value": date_of_birth,
        "coordinates": date_of_birth_coords,
        "score": date_of_birth_score
    }

    gen_dict = {
        "name": OCRFieldNames.GENDER,
        "value": gender_field,
        "coordinates": gender_field_coords,
        "score":gender_field_score
    }

    first_name_dict = {
        "name": OCRFieldNames.FIRST_NAME,
        "value": first_name,
        "coordinates":first_name_coords,
        "score":first_name_score
    }

    middle_name_dict = {
        "name": OCRFieldNames.MIDDLE_NAME,
        "value": middle_name,
        "coordinates": middle_name_coords,
        "score": middle_name_score
    }

    last_name_dict = {
        "name": OCRFieldNames.LAST_NAME,
        "value": last_name,
        "coordinates": last_name_coords,
        "score": last_name_score
    }

    id_number_dict = id_number_extraction(front_ocr_results)

    return [first_name_dict, middle_name_dict, last_name_dict, dob_dict, gen_dict, id_number_dict]


def id_number_extraction(image_back_ocr_results):
    pattern = r'^\d{1,6}/\d{1,2}/\d+$'

    results = []
    for coords, text in image_back_ocr_results:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
        cent_y = (y1 + y2 + y3 + y4) / 4
        cent_x = (x1 + x2 + x3 + x4) / 4
        results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })
    sorted_results = sorted(results, key=lambda x: x['cent_y'])

    txts = [line[1][0] for line in image_back_ocr_results]
    id_number = id_number_score = id_number_coords = None
    possible_id_number = []

    for i in txts:
        matches = re.search(pattern, i)
        if matches is not None:
            possible_id_number.append(i)
    
    if id_number is None:
        for i in txts:
            if '/' not in i:
                possible_id_number.append(i)

    for id in possible_id_number:
        j = "".join([i for i in id if i != '/'])
        if len(j) == 9:
            for d in sorted_results:
                if(d['text'][0]==id):
                    id_number_score = d['text'][1]
                    id_number_coords = d['coords']
            return {
                'name': OCRFieldNames.ID_NUMBER,
                'value': j,
                'coordinates': id_number_coords,
                'score': id_number_score
            }
    return {
        'name': OCRFieldNames.ID_NUMBER,
        'value': None,
        'coordinates': None,
        'score': None
    }