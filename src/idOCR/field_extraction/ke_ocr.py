import re
import random
import fuzzysearch

class OCRFieldNames:
    ID_NUMBER = "id_number"
    FIRST_NAME = "first_name"
    MIDDLE_NAME = "middle_name"
    LAST_NAME = "last_name"
    DATE_OF_BIRTH = "date_of_birth"
    GENDER = "gender"

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

def concat(text_blocks):
    '''
    Concatenate the text from the blocks and calculate the average score.
    TODO: add bbox concatenation
    '''
    if not text_blocks:
        return ('', 0.0)
    # Concatenate the text from the blocks
    concatenated_text = ''
    score = 0.0
    for block in text_blocks:
        if concatenated_text == '':
            concatenated_text = block['text'][0]
        else:
            concatenated_text += ' ' + block['text'][0]
        score += block['text'][1]
    score = score / len(text_blocks)
    concatenated_text = concatenated_text.strip(' ')

    return concatenated_text, score

def break_blocks_per_word(text_blocks):
    new_blocks = []
    for block in text_blocks:
        block_text = block['text'][0]
        block_score = block['text'][1]
        block_coords = block['coords']
        words = block_text.split(' ')
        for word in words:
            if word:
                # Create a new block for each word
                new_block = {
                    'text': (word, block_score),
                    'coords': block_coords
                }
                new_blocks.append(new_block)

    return new_blocks


def is_same_line(current_block, next_block):
    (x1_1, y1_1), (x2_1, y2_1), (x3_1, y3_1), (x4_1, y4_1) = current_block['coords']
    cent_y_right_current = (y2_1 + y3_1) / 2
    cent_y_left_current = (y1_1 + y4_1) / 2
    (x1_2, y1_2), (x2_2, y2_2), (x3_2, y3_2), (x4_2, y4_2) = next_block['coords']
    cent_y_right_next = (y2_2 + y3_2) / 2
    cent_y_left_next = (y1_2 + y4_2) / 2
    return (y1_2 < cent_y_right_current < y4_2) and (y2_1 < cent_y_left_next < y3_1)


def concatenate_names_on_same_line(text_blocks):
    if not text_blocks:
        return []

    merged_texts = []
    visited = set()

    for i in range(len(text_blocks)):
        if i in visited:
            continue

        current_text = text_blocks[i]['text'][0]
        current_block = text_blocks[i]
        visited.add(i)

        for j in range(i + 1, len(text_blocks)):
            next_block = text_blocks[j]

            if is_same_line(current_block, next_block):
                current_text += " " + next_block['text'][0]
                visited.add(j)  # Mark this block as visited

        merged_texts.append(current_text)
    return merged_texts

def concatenate_blocks_on_same_line(text_blocks):
    if not text_blocks:
        return []

    merged_blocks = []
    visited = set()

    for i in range(len(text_blocks)):
        if i in visited:
            continue

        current_block = text_blocks[i]
        same_line_blocks = [current_block]

        visited.add(i)

        for j in range(i + 1, len(text_blocks)):
            next_block = text_blocks[j]

            if is_same_line(current_block, next_block):
                same_line_blocks.append(next_block)
                visited.add(j)  # Mark this block as visited

        merged_blocks.append(same_line_blocks)
    return merged_blocks


def identify_new_card(text_list):
    text_list = [[boxes, (text, float(score))] for boxes, (text, score) in text_list]
       
    for item in text_list:
        if (len(fuzzysearch.find_near_matches('nationalidentitycard', item[1][0].replace(' ','').lower(), max_l_dist=4))>=1) or \
        (len(fuzzysearch.find_near_matches('kitambulishochataifa', item[1][0].replace(' ','').lower(), max_l_dist=4))>=1):
            return 1
    return 0



class IdFieldExtractionRules:

    def __init__(self) -> None:
        self.id_number_idx = None
        self.fullname_idx = None
        self.dob_idx = None
        self.gender_idx = None

    @staticmethod
    def sort_detections(ocr_detections):
        ### Sort OCR results using centroid-y of detected polygons
        detections = []
        for coords, text in ocr_detections:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
            cent_y = (y1 + y2 + y3 + y4) / 4
            cent_x = (x1 + x2 + x3 + x4) / 4
            detections.append({
                'cent_x': cent_x,
                'cent_y': cent_y,
                'coords': coords,
                'text': text
            })
        return detections
        # return sorted(detections, key=lambda x: x['cent_y'])

    @staticmethod
    def is_substring(substring, string):
        i, j = 0, 0
        while i < len(substring) and j < len(string):
            if substring[i] == string[j]:
                i += 1
            j += 1
        return i == len(substring)

    def extract_id_number(self):
        republic_of_kenya_idx = None
        jamhuri_ya_kenya_idx = None

        # Find the index of jamhuri_ya_kenya or republic_of_kenya
        for idx, detection in enumerate(self.ocr_detections):

            text = detection['text'][0].replace(' ', '').lower()

            # Get position of republic of kenya
            republic_of_kenya_matches = fuzzysearch.find_near_matches('icofkenya', text, max_l_dist=2)
            if republic_of_kenya_matches:
                republic_of_kenya_idx = idx

            # Get position of republic of kenya
            jamhuri_ya_kenya_pos = fuzzysearch.find_near_matches('jamhuriya', text, max_l_dist=2)
            if jamhuri_ya_kenya_pos:
                jamhuri_ya_kenya_idx = idx

        # Get the matched numbers
        start_index = republic_of_kenya_idx if republic_of_kenya_idx is not None else jamhuri_ya_kenya_idx
        if start_index is None:
            pass
        else:
            matched_numbers = []
            idx = start_index
            while idx < min(start_index + 10, len(self.ocr_detections)):
                text = self.ocr_detections[idx]['text'][0].replace(' ', '')

                # Check for number matches
                number_pattern = r'\d{7,}'
                number_matches = re.search(number_pattern, text)

                if number_matches:
                    matched_numbers.append({'idx': idx, 'value': number_matches.group(0)})
                    if len(matched_numbers) == 2:
                        break
                idx += 1

        # Identify which of these is the actual ID Number
        id_number_value = None
        id_number_score = 0.0

        if jamhuri_ya_kenya_idx is not None:
            (coj_x1, coj_y1), (coj_x2, coj_y2), (coj_x3, coj_y3), (coj_x4, coj_y4) = \
                self.ocr_detections[jamhuri_ya_kenya_idx]['coords']

            for matched_number in matched_numbers:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.ocr_detections[matched_number['idx']]['coords']
                cent_x = (x1 + x2 + x3 + x4) / 4
                if cent_x > coj_x2:
                    id_number_value = matched_number['value']
                    id_number_score = self.ocr_detections[matched_number['idx']]['text'][1]
                    self.id_number_idx = matched_number['idx']

        if republic_of_kenya_idx is not None:
            (cor_x1, cor_y1), (cor_x2, cor_y2), (cor_x3, cor_y3), (cor_x4, cor_y4) = \
                self.ocr_detections[republic_of_kenya_idx]['coords']

            for matched_number in matched_numbers:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.ocr_detections[matched_number['idx']]['coords']
                cent_x = (x1 + x2 + x3 + x4) / 4
                if cent_x > cor_x1:
                    id_number_value = matched_number['value']
                    id_number_score = self.ocr_detections[matched_number['idx']]['text'][1]
                    self.id_number_idx = matched_number['idx']

        return ('', id_number_score) if id_number_value is None else (id_number_value, id_number_score)


    def extract_date_of_birth(self):

        date_pattern = r'^(0?[0-9]|[12][0-9]|3[01])[-/\.,:]?(0?[0-9]|1[012])[-/\.,:]?(19[0-9]{2}|49[0-9]{2}|20[0-9]{2}|[0-9]{2})$'
        dob_value = None
        dob_score = 0.0

        idx = self.id_number_idx + 1 if self.id_number_idx is not None else 0
        while idx < len(self.ocr_detections):
            text = self.ocr_detections[idx]['text'][0].replace(' ', '')
            date_matches = re.search(date_pattern, text)
            if date_matches:
                dob_value = date_matches
                dob_score = self.ocr_detections[idx]['text'][1]
                self.dob_idx = idx
                break
            idx += 1

        if dob_value:
            if len(dob_value.group(0)) == 4:
                return (f'{dob_value.group(0)}-00-00', dob_score)
            else:
                year = dob_value.group(3)
                month = dob_value.group(2)
                day = dob_value.group(1)
                if year.startswith('49'):
                    year = '19' + year[2:]
                return (f'{year}-{month}-{day}', dob_score)

        return ('', dob_score) if dob_value is None else (dob_value, dob_score)


    def extract_gender(self):

        gender_value = None
        gender_score = 0.0
        female_substrings = ['femal', 'fehale', 'eemal', 'ffmal', 'fewal', 'fehawe']
        for idx, detection in enumerate(self.ocr_detections):

            text = detection['text'][0].replace(' ', '').lower()
            score = detection['text'][1]

            # Get position of republic of kenya
            for substring in female_substrings:
                if self.is_substring(substring, text):
                    gender_value = 'FEMALE'
                    gender_score = score
                    self.gender_idx = idx
                    break

        return (gender_value, gender_score) if gender_value else ("MALE", random.uniform(0.9,1))

    def extract_fullname(self):

        possible_fullnames = []
        skip_words_approx = ['fullname', 'number', 'dateof', 'jamhuriya', 'ofkenya', 'placeof']
        skip_words_exact = ['sex', 'male', 'female']

        start = self.id_number_idx
        end = self.dob_idx
        for detection in self.ocr_detections[start:end]:

            text = detection['text'][0].replace(' ', '').lower()

            skip = False
            for word in skip_words_approx:
                if fuzzysearch.find_near_matches(word, text, max_l_dist=1):
                    skip = True
                    break
            for word in skip_words_exact:
                if text == word:
                    skip = True
                    break
            if skip:
                continue

            # Check for number matches
            name_pattern = r'([a-zA-Z]\s?)+'
            name_matches = re.search(name_pattern, text)
            if name_matches:
                possible_fullnames.append(detection)

        # Combining the full name values which are side by side
        concatenated_blocks = concatenate_blocks_on_same_line(possible_fullnames)

        # Get fullname value
        fullname_block = None
        for idx_block in concatenated_blocks:
            curr_text, _ = concat(idx_block)
            fullname, _ =  concat(fullname_block)
            if len(curr_text) > len(fullname):
                fullname_block = idx_block

        first_name = ('', 0.0)
        middle_name = ('', 0.0)
        last_name = ('', 0.0)

        if fullname_block:
            #break the fullname block further according to the spaces
            fullname_block = break_blocks_per_word(fullname_block)
            if len(fullname_block) > 3:
                first_name = concat([fullname_block[0]])
                middle_name = concat(fullname_block[1:-1])
                last_name = concat([fullname_block[-1]])

            elif len(fullname_block) == 3:
                first_name = concat([fullname_block[0]])
                middle_name = concat([fullname_block[1]])
                last_name = concat([fullname_block[-1]])

            elif len(fullname_block) == 2:
                first_name = concat([fullname_block[0]])
                last_name = concat([fullname_block[-1]])
            else:
                first_name = concat([fullname_block[0]])

        return first_name, middle_name, last_name

    def extract_fields(self, ocr_detections):

        demographic_fields = [
            { "name": OCRFieldNames.FIRST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.LAST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.MIDDLE_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.GENDER, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.DATE_OF_BIRTH, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.ID_NUMBER, "value": None,"coordinates": None,"score": None}
    ]


        self.ocr_detections = self.sort_detections(ocr_detections)
        gender_value,gender_score = self.extract_gender()
        id_number_value,id_number_score = self.extract_id_number()
        date_of_birth_value, date_of_birth_score = self.extract_date_of_birth()
        first_name, middle_name, last_name = self.extract_fullname()

        first_name_value, first_name_score = first_name
        middle_name_value, middle_name_score = middle_name
        last_name_value, last_name_score = last_name

        demographic_fields = [
            { "name": OCRFieldNames.FIRST_NAME, "value": first_name_value,"coordinates": None,"score": first_name_score},
            { "name": OCRFieldNames.LAST_NAME, "value": last_name_value,"coordinates": None,"score": last_name_score},
            { "name": OCRFieldNames.MIDDLE_NAME, "value":middle_name_value,"coordinates": None,"score":middle_name_score},
            { "name": OCRFieldNames.GENDER, "value": gender_value,"coordinates": None,"score": gender_score},
            { "name": OCRFieldNames.DATE_OF_BIRTH, "value":date_of_birth_value,"coordinates": None,"score": date_of_birth_score},
            { "name": OCRFieldNames.ID_NUMBER, "value": id_number_value,"coordinates": None,"score": id_number_score}
    ]

        return demographic_fields



class IdFieldExtractionRulesNewCard:

    def __init__(self) -> None:
        self.id_number_idx = None
        self.fullname_idx = None
        self.dob_idx = None
        self.gender_idx = None

    @staticmethod
    def sort_detections(ocr_detections):

        detections = []
        for coords, text, confidence in ocr_detections:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
            cent_y = (y1 + y2 + y3 + y4) / 4
            cent_x = (x1 + x2 + x3 + x4) / 4
            score = confidence
            detections.append({
                'cent_x': cent_x,
                'cent_y': cent_y,
                'coords': coords,
                'text': text,
                'score': score
            })
        return detections
        # return sorted(detections, key=lambda x: x['cent_y'])

    @staticmethod
    def is_substring(substring, string):
        i, j = 0, 0
        while i < len(substring) and j < len(string):
            if substring[i] == string[j]:
                i += 1
            j += 1
        return i == len(substring)

    def extract_id_number(self):
        republic_of_kenya_idx = None
        jamhuri_ya_kenya_idx = None

        for idx, detection in enumerate(self.ocr_detections):

            text = detection['text'].replace(' ', '').lower()

            republic_of_kenya_matches = fuzzysearch.find_near_matches('icofkenya', text, max_l_dist=2)
            if republic_of_kenya_matches:
                republic_of_kenya_idx = idx

            jamhuri_ya_kenya_pos = fuzzysearch.find_near_matches('jamhuriya', text, max_l_dist=2)
            if jamhuri_ya_kenya_pos:
                jamhuri_ya_kenya_idx = idx

        start_index = republic_of_kenya_idx if republic_of_kenya_idx is not None else jamhuri_ya_kenya_idx
        if start_index is None:
            pass
        else:
            matched_numbers = []
            idx = start_index
            while idx < min(start_index + 20, len(self.ocr_detections)):
                text = self.ocr_detections[idx]['text'].replace(' ', '')

                number_pattern = r'\d{7,}'
                number_matches = re.search(number_pattern, text)
                if number_matches:
                    matched_numbers.append({'idx': idx, 'value': number_matches.group(0)})
                    if len(matched_numbers) == 2:
                        break
                idx += 1


        id_number_value = None
        id_number_score = 0.0

        if jamhuri_ya_kenya_idx is not None:
            (coj_x1, coj_y1), (coj_x2, coj_y2), (coj_x3, coj_y3), (coj_x4, coj_y4) = \
                self.ocr_detections[jamhuri_ya_kenya_idx]['coords']

            for matched_number in matched_numbers:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.ocr_detections[matched_number['idx']]['coords']
                cent_x = (x1 + x2 + x3 + x4) / 4
                if cent_x > coj_x2:
                    id_number_value = matched_number['value']
                    id_number_score = self.ocr_detections[matched_number['idx']]['score']
                    self.id_number_idx = matched_number['idx']

        if republic_of_kenya_idx is not None:
            (cor_x1, cor_y1), (cor_x2, cor_y2), (cor_x3, cor_y3), (cor_x4, cor_y4) = \
                self.ocr_detections[republic_of_kenya_idx]['coords']

            for matched_number in matched_numbers:
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.ocr_detections[matched_number['idx']]['coords']
                cent_x = (x1 + x2 + x3 + x4) / 4
                if cent_x > cor_x1:
                    id_number_value = matched_number['value']
                    id_number_score = self.ocr_detections[matched_number['idx']]['score']
                    self.id_number_idx = matched_number['idx']

        return ('', id_number_score) if id_number_value is None else (id_number_value, id_number_score)

    def extract_date_of_birth(self):
        date_pattern = r'^(0?[1-9]|[12][0-9]|3[01])[-/\.,]?(0?[1-9]|1[012])[-/\.,]?(19[0-9]{2}|20[0-9]{2}|[0-9]{2})$'
        dob_value = None
        dob_score = 0.0
        best_match = None
        highest_cent_x = -float('inf')

        idx = 0
        while idx < len(self.ocr_detections):
            text = self.ocr_detections[idx]['text'].replace(' ', '')
            date_matches = re.search(date_pattern, text)
            if date_matches:
                if self.ocr_detections[idx]['cent_x'] > highest_cent_x:
                    highest_cent_x = self.ocr_detections[idx]['cent_x']
                    dob_value = date_matches
                    dob_score = self.ocr_detections[idx]['score']
                    self.dob_idx = idx
                    best_match = self.ocr_detections[idx]
            idx += 1

        if dob_value:
            if len(dob_value.group(0)) == 4:
                return (f'{dob_value.group(0)}-00-00', dob_score)
            else:
                return (f'{dob_value.group(3)}-{dob_value.group(2)}-{dob_value.group(1)}', dob_score)

        return ('', dob_score)

    def extract_gender(self):

        gender_value = None
        gender_score = 0.0
        female_substrings = ['femal', 'fehale', 'eemal', 'ffmal', 'fewal', 'fehawe']
        for idx, detection in enumerate(self.ocr_detections):

            text = detection['text'].replace(' ', '').lower()
            score = detection['score']

            # Get position of republic of kenya
            for substring in female_substrings:
                if self.is_substring(substring, text):
                    gender_value = 'FEMALE'
                    gender_score = score
                    self.gender_idx = idx
                    break

        return (gender_value, gender_score) if gender_value else ("MALE", random.uniform(0.9,1))

    def extract_fullname(self,raw_detections):

        start = None
        end = self.dob_idx if self.dob_idx is not None else 15
        national_id_label = None
        national_id_label_idx = None
        national_id_coords = None
        height = None
        last_name = ''
        first_name = ''
        middle_name = ''
        first_name_box = None
        headers = ['jamhuriyakenya','republicofkenya']

        for idx,detection in enumerate(raw_detections[start:end]):
            text = detection[1].replace(' ', '').lower()
            if fuzzysearch.find_near_matches(text.lower(), "nationalidentitycard", max_l_dist=4):
                national_id_label_idx = idx
                coords = raw_detections[idx][0]
                national_id_coords = coords
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
                height = min(abs(y2-y3),abs(y4-y1))
                start = idx
                break

        for detection in raw_detections[start:end]:
            temp_coord = detection[0]
            text = detection[1]
            score = detection[2]
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = temp_coord
            (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = national_id_coords
            temp_height = min(abs(y2-y3),abs(y4-y1))
            if text.lower().replace(' ','') not in headers:
                if temp_height > height and X1>x1:
                    if last_name == '':
                            last_name = (detection[1], score)
                    else:
                        if first_name == '':
                            first_name = (detection[1], score)
                            first_name_box = detection[0]

                        else:
                            (x1_, y1_), (x2_, y2_), (x3_, y3_), (x4_, y4_) = first_name_box
                            (x1__, y1__), (x2__, y2__), (x3__, y3__), (x4__, y4__) = detection[0]
                            if abs(y1__ - y1_) < 5:
                                middle_name = (detection[1], score)

        return first_name, middle_name, last_name

    
    def extract_fields(self, ocr_detections):

        demographic_fields = [
            { "name": OCRFieldNames.FIRST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.LAST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.MIDDLE_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.GENDER, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.DATE_OF_BIRTH, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.ID_NUMBER, "value": None,"coordinates": None,"score": None}]


        raw_detections = ocr_detections
        self.ocr_detections = self.sort_detections(ocr_detections)
        gender_value,gender_score = self.extract_gender()
        id_number_value,id_number_score = self.extract_id_number()
        date_of_birth_value, date_of_birth_score = self.extract_date_of_birth()
        first_name, middle_name, last_name = self.extract_fullname()

        first_name_value, first_name_score = first_name
        middle_name_value, middle_name_score = middle_name
        last_name_value, last_name_score = last_name

        demographic_fields = [
            { "name": OCRFieldNames.FIRST_NAME, "value": first_name_value,"coordinates": None,"score": first_name_score},
            { "name": OCRFieldNames.LAST_NAME, "value": last_name_value,"coordinates": None,"score": last_name_score},
            { "name": OCRFieldNames.MIDDLE_NAME, "value":middle_name_value,"coordinates": None,"score":middle_name_score},
            { "name": OCRFieldNames.GENDER, "value": gender_value,"coordinates": None,"score": gender_score},
            { "name": OCRFieldNames.DATE_OF_BIRTH, "value":date_of_birth_value,"coordinates": None,"score": date_of_birth_score},
            { "name": OCRFieldNames.ID_NUMBER, "value": id_number_value,"coordinates": None,"score": id_number_score}
    ]

        return demographic_fields


def extract_kyc_fields(ocr_detections,back_detections=None, image=None):
    ocr_detections = [ [boxes, (text, float(score))] for boxes, (text, score) in ocr_detections]

    demographic_fields = [
            { "name": OCRFieldNames.FIRST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.LAST_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.MIDDLE_NAME, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.GENDER, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.DATE_OF_BIRTH, "value": None,"coordinates": None,"score": None},
            { "name": OCRFieldNames.ID_NUMBER, "value": None,"coordinates": None,"score": None}]

    # Determine ID type using OCR
    id_type = get_id_type_by_ocr(ocr_detections)
    if id_type != "National ID":  # If not a National ID, return empty fields
        return demographic_fields

    if identify_new_card(ocr_detections):
        id_field_extraction_rules = IdFieldExtractionRulesNewCard() 
    else:
        id_field_extraction_rules = IdFieldExtractionRules()

    extracted_fields = id_field_extraction_rules.extract_fields(ocr_detections)

    return extracted_fields