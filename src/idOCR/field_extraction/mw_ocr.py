import pandas as pd
import numpy as np
from datetime import datetime
import logging
import re
from dateutil.parser import parse
from Levenshtein import distance as levenshtein_distance
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
        return "Other ID"
        
        
    except Exception as e:
        logger.error(f"Error in OCR ID type detection: {str(e)}")
        return None
    
        

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
          "nov", "dec"]


def y_lower(res):
    lowermost_y = min(x[1] for x in res)
    return lowermost_y


def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def check_expiry_text(text):
    if (
            "date" in text or "cate" in text or "dae" in text or "dat" in text or "dme" in text) and (
            "exp" in text or "xpiry" in text):
        return True
    if "exp" in text and " " in text:
        return True
    return False


def check_issue_text(text):
    if (
            "date" in text or "cate" in text or "dae" in text or "dat" in text) and (
            "iss" in text or "issue" in text):
        return True
    if "iss" in text and " " in text:
        return True

    return False

def check_dob_text(text):
    if (
            "date" in text or "cate" in text or "dae" in text or "dat" in text) and (
            "birth" in text or "bint" in text):
        return True
    if "birth" in text and " " in text:
        return True
    return False

def check_lastname_text(text):
    if (
            "bambo" in text or "banbo" in text or 'bambu' in text or "bambe" in text or
            "surname" in text or "sunam" in text or "sumane" in text or "sumame" in text or "sumam" in text or "surmam" in text or "surnam" in text):
        return True

    return False


def check_firstname_text(text):
    #     print("text in check first name: ", text)
    if (
            "maina" in text or "ena/" in text or "other name" in text or 'other nane' in text or 'oter nane' in text
            or 'oter' in text or 'dzina' in text or ('other' in text and 'name' in text)) and (
            "bambo" not in text and "banbo" not in text
            and 'bambu' not in text
            and "surname" not in text
            and "sumane" not in text and "sumame" not in text
            and "sumam" not in text and "surmam" not in text
            and "surnam" not in text):
        return True

    return False

def check_lastname(text):
    text = re.sub('[^A-Za-z0-9]+', '', text)

    return text

def check_id_no_text(text):
    fuzzy_res = fuzzysearch.find_near_matches('identification', text.lower(), max_l_dist=3)

    if len(fuzzy_res) > 0 and ('citizen' not in text and 'citzen' not in text and "citien" not in text
                               and "citiz" not in text and "chiphaso" not in text and 'chip' not in text):
        return True

    return False


def count_words(text):
    return len(text.split())

def check_gender_text(text):
    if (
            "sex" == text or "ser" == text or 'ses' == text):
        return True

    return False

def check_gender_upper_block(text):
    if ("mwamuna" in text or "mkazi" in text):
        return True

    return False

def process_extracted_gender(text):
    text = re.sub('[^A-Za-z0-9]+', '', text)
    if text == 'M' or text == 'F':
        return text
    else:
        return None


def check_rep_of_mw_text(text):
    if (
            "republic" in text or "malawi" in text):
        return True

    return False


def match_MW1_text(text):
    if text == "mw1" or text == 'mwi' or text == 'mw':
        return True
    else:
        return False


def extract_date(date_text):
    try:
        if date_text is None or pd.isnull(date_text):
            return None

        ## Replace special characters with space and multiple spaces with single space
        date_text = re.sub('[^A-Za-z0-9]+', ' ', date_text)
        date_text = re.sub("\s\s+", " ", date_text)
        date_text = date_text.lower()

        numbers = re.findall(r'\d+',date_text)
        month = re.sub(r'\d+', '', date_text).strip()

        if len(numbers)!=2:
            return None

        if month == "" or len(month)!=3:
            return None

        if month.lower() in months:
            date = "{} {} {}".format(numbers[0],month,numbers[1])
            return date

        else:
            potential_months = list()
            for val in months:
                if levenshtein_distance(month,val) <=1:
                    potential_months.append(val)

            if len(potential_months)>1 or len(potential_months)==0:
                return None
            else:
                month = potential_months[0]
                date = "{} {} {}".format(numbers[0],month,numbers[1])
                return date
        return date
    except Exception as e:
        print("Unable to parse date for {}. Error {}".format(date_text,str(e)))


def refine_date_predictions(date_text):
    if pd.isnull(date_text) or date_text is None:
        return None

    splitted_date = date_text.split(" ")
    day = int(splitted_date[0])
    year = int(splitted_date[2])
    if day > 31 or day < 1:
        return None
    if len(str(year)) != 4 or year > 2100 or year<1900:
        return None
    return date_text



DEFAULT_RES = (None, 0.0)

def extract_kyc_fields(ocr_detections,back_detections=None, image=None):

    ocr_detections = [ [boxes, (text, float(score))] for boxes, (text, score) in ocr_detections]

    demographic_fields = [
        { "name": OCRFieldNames.FIRST_NAME, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.LAST_NAME, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.MIDDLE_NAME, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.GENDER, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.DATE_OF_BIRTH, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.ID_NUMBER, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.DATE_OF_ISSUE, "value": None,"coordinates": None,"score": None},
        { "name": OCRFieldNames.DATE_OF_EXPIRY, "value": None,"coordinates": None,"score": None}
    ]


    id_type=get_id_type_by_ocr(ocr_detections)

    if id_type != "National ID":
        return demographic_fields

    def extract_date_of_expiry(result):
        try:
            # result = ocr.ocr(image, cls=True)
            date_of_expiry_value_block = None
            date_of_expiry_block = None
            diff = None

            ## Extract using Date of Expiry
            for val in result:
                text = val[1][0].lower()
                if date_of_expiry_block is None:
                    if check_expiry_text(text):
                        date_of_expiry_block = val
                        lowermost_y_doe = y_lower(date_of_expiry_block[0])
                        leftmost_x_doe = min(x[0] for x in date_of_expiry_block[0])
                        rightmost_x_doe = max(x[0] for x in date_of_expiry_block[0])
                        break
            if date_of_expiry_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_doe and \
                                rightmost_x_cur > leftmost_x_doe and \
                                rightmost_x_doe > leftmost_x_cur and has_numbers(
                                text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_expiry_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doe)
                    else:
                        if lowermost_y_cur > lowermost_y_doe and abs(
                                lowermost_y_cur - lowermost_y_doe) < diff and \
                                rightmost_x_cur > leftmost_x_doe and\
                                rightmost_x_doe > leftmost_x_cur and has_numbers(
                                text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_expiry_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doe)

                if date_of_expiry_value_block is not None:
                    # print("Extracted by comparing date of expiry")
                    return date_of_expiry_value_block

            ## Date of Expiry cannot be extracted, extract using date of issue
            date_of_issue_block = None
            for val in result:
                text = val[1][0].lower()
                if date_of_issue_block is None:
                    if check_issue_text(text):
                        date_of_issue_block = val
                        lowermost_y_doi = y_lower(date_of_issue_block[0])
                        leftmost_x_doi = min(x[0] for x in date_of_issue_block[0])
                        rightmost_x_doi = max(x[0] for x in date_of_issue_block[0])
                        break

            if date_of_issue_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()

                    if diff is None:
                        if lowermost_y_cur > lowermost_y_doi and rightmost_x_cur > rightmost_x_doi and leftmost_x_cur > rightmost_x_doi and has_numbers(
                                text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_expiry_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doi)

                if date_of_expiry_value_block is not None:
                    # print("Extracted by comparing date of issue")
                    return date_of_expiry_value_block

            ## Unable to Find even after date of issue, extract all dates and find
            date_blocks = dict()
            for val in result:
                text = val[1][0].lower()
                date_val = extract_date(text)
                refined_date_value = refine_date_predictions(date_val)
                if refined_date_value is not None:
                    ## Extract year from date and checking if length of year is 4
                    year = date_val.split(" ")[-1].strip()
                    if len(year) != 4:
                        continue
                    date_blocks[int(year)] = val

            if len(date_blocks) != 3:
                # print("Different than 3 date blocks extracted.Unable to extract")
                return None
            max_year = max([y for y in date_blocks.keys()])
            date_of_expiry_value_block = date_blocks[max_year]
            if date_of_expiry_value_block is not None:
                # print("Extracted by comparing year of all dates")
                return date_of_expiry_value_block

        except Exception as e:
            # print(str(e))
            pass
        return None

    def process_date_of_expiry(result):
        try:
            res = extract_date_of_expiry(result)
            if res is None:
                return DEFAULT_RES
            res_date = extract_date(res[1][0].lower())
            res_score = res[1][1]
            refined_date = refine_date_predictions(res_date)
            refined_date_formatted = None
            if refined_date is not None:
                # refined_date_formatted = datetime.strptime(
                #     refined_date, '%d %b %Y').strftime("%Y-%m-%d")
                refined_expiry_date_formatted = datetime.strptime(
                    refined_date, '%d %b %Y')

            return refined_expiry_date_formatted, res_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    def extract_date_of_issue(result):
        try:
            date_of_issue_value_block = None
            date_of_issue_block = None
            diff = None

            ## Extract using Date of Expiry
            for val in result:
                text = val[1][0].lower()
                if date_of_issue_block is None:
                    if check_issue_text(text):
                        # print("doi text: ", text)
                        date_of_issue_block = val
                        lowermost_y_doi = y_lower(date_of_issue_block[0])
                        leftmost_x_doi = min(x[0] for x in date_of_issue_block[0])
                        rightmost_x_doi = max(x[0] for x in date_of_issue_block[0])
                        break
            if date_of_issue_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_doi and \
                                rightmost_x_cur > leftmost_x_doi and \
                                rightmost_x_doi > leftmost_x_cur and has_numbers(
                            text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_issue_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doi)
                    else:
                        if lowermost_y_cur > lowermost_y_doi and abs(
                                lowermost_y_cur - lowermost_y_doi) < diff and \
                                rightmost_x_cur > leftmost_x_doi and \
                                rightmost_x_doi > leftmost_x_cur and has_numbers(
                            text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_issue_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doi)

                if date_of_issue_value_block is not None:
                    # print("Extracted by comparing date of issue")
                    return date_of_issue_value_block

            ## Date of Issue cannot be extracted, extract using date of Expiry
            date_of_expiry_block = None
            for val in result:
                text = val[1][0].lower()
                if date_of_expiry_block is None:
                    if check_expiry_text(text):
                        # print("doe text: ", text)
                        date_of_expiry_block = val
                        lowermost_y_doe = y_lower(date_of_expiry_block[0])
                        leftmost_x_doe = min(x[0] for x in date_of_expiry_block[0])
                        rightmost_x_doe = max(x[0] for x in date_of_expiry_block[0])
                        break

            if date_of_expiry_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()

                    if diff is None:
                        #                     if lowermost_y_cur > lowermost_y_doi and rightmost_x_cur > rightmost_x_doi and leftmost_x_cur > rightmost_x_doi and has_numbers(
                        #                             text):
                        if lowermost_y_cur < lowermost_y_doe and rightmost_x_cur < rightmost_x_doe and leftmost_x_cur < rightmost_x_doe and has_numbers(
                                text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_issue_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_doi)

                if date_of_issue_value_block is not None:
                    # print("Extracted by comparing date of expiry")
                    return date_of_issue_value_block

            ## Unable to Find even after date of expiry, extract all dates and find
            date_blocks = dict()
            for val in result:
                text = val[1][0].lower()
                date_val = extract_date(text)
                refined_date_value = refine_date_predictions(date_val)
                if refined_date_value is not None:
                    ## Extract year from date and checking if length of year is 4
                    year = date_val.split(" ")[-1].strip()
                    if len(year) != 4:
                        continue
                    date_blocks[int(year)] = val

            if len(date_blocks) != 3:
                # print("Different than 3 date blocks extracted.Unable to extract")
                return None
            max_year = max([y for y in date_blocks.keys()])
            min_year = min([y for y in date_blocks.keys()])
            doi_year = [x for x in date_blocks.keys() if x not in [min_year, max_year]][0]

            date_of_issue_value_block = date_blocks[doi_year]
            if date_of_issue_value_block is not None:
                # print("Extracted by comparing year of all dates")
                return date_of_issue_value_block

        except Exception as e:
            # print(str(e))
            pass
        return None

    def process_date_of_issue(result):
        try:
            res = extract_date_of_issue(result)
            if res is None:
                return DEFAULT_RES
            res_date = extract_date(res[1][0].lower())
            res_score = res[1][1]
            refined_date = refine_date_predictions(res_date)
            refined_issue_date_formatted = datetime.strptime(
                    refined_date, '%d %b %Y')
            return refined_issue_date_formatted, res_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    def extract_dob(result):
        try:
            date_of_birth_value_block = None
            date_of_birth_block = None
            diff = None

            ## Extract using Date of Birth Block
            for val in result:
                text = val[1][0].lower()
                if date_of_birth_block is None:
                    if check_dob_text(text):
                        date_of_birth_block = val
                        lowermost_y_dob = y_lower(date_of_birth_block[0])
                        leftmost_x_dob = min(x[0] for x in date_of_birth_block[0])
                        rightmost_x_dob = max(x[0] for x in date_of_birth_block[0])
                        break
            if date_of_birth_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_dob and \
                                rightmost_x_cur > leftmost_x_dob and \
                                rightmost_x_dob > leftmost_x_cur and has_numbers(
                            text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_birth_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_dob)
                    else:
                        if lowermost_y_cur > lowermost_y_dob and abs(
                                lowermost_y_cur - lowermost_y_dob) < diff and \
                                rightmost_x_cur > leftmost_x_dob and \
                                rightmost_x_dob > leftmost_x_cur and has_numbers(
                            text):
                            extracted_date = extract_date(val[1][0].lower())
                            if extracted_date is not None:
                                date_of_birth_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_dob)

                if date_of_birth_value_block is not None:
                    # print("Extracted by comparing date of birth")
                    return date_of_birth_value_block

            ## Unable to Find even after date of issue, extract all dates and find
            date_blocks = dict()
            for val in result:
                text = val[1][0].lower()
                date_val = extract_date(text)
                refined_date_value = refine_date_predictions(date_val)
                if refined_date_value is not None:
                    ## Extract year from date and checking if length of year is 4
                    year = date_val.split(" ")[-1].strip()
                    if len(year) != 4:
                        continue
                    date_blocks[int(year)] = val

            if len(date_blocks) != 3:
                # print("Different than 3 date blocks extracted.Unable to extract")
                return None
            min_year = min([y for y in date_blocks.keys()])
            date_of_birth_value_block = date_blocks[min_year]
            if date_of_birth_value_block is not None:
                # print("Extracted by comparing year of all dates")
                return date_of_birth_value_block

        except Exception as e:
            # print(str(e))
            pass

        return None

    def process_date_of_birth(result):
        try:
            res = extract_dob(result)

            if res is None:
                return DEFAULT_RES

            dob = extract_date(res[-1][0].lower())
            dob_score = res[-1][1]

            dob_formatted = datetime.strptime(
                dob, '%d %b %Y')

            return dob_formatted, dob_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    def extract_firstname(result):
        try:
            firstname_value_block = None
            firstname_block = None
            diff = None
            ## Extract using Surname
            for val in result:
                text = val[1][0].lower()
                if firstname_block is None:

                    if check_firstname_text(text):
                        firstname_block = val
                        topmost_y_fname = max(x[1] for x in firstname_block[0])
                        lowermost_y_fname = y_lower(firstname_block[0])
                        leftmost_x_fname = min(x[0] for x in firstname_block[0])
                        rightmost_x_fname = max(x[0] for x in firstname_block[0])
                        break
            if firstname_block is not None:
                for val in result:
                    topmost_y_cur = max(x[1] for x in val[0])
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_fname and topmost_y_cur > topmost_y_fname and not has_numbers(
                                text):
                            extracted_fname = text
                            if extracted_fname is not None:
                                firstname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_fname)
                    else:
                        if lowermost_y_cur > lowermost_y_fname and topmost_y_cur > topmost_y_fname and abs(
                                lowermost_y_cur - lowermost_y_fname) < diff and not has_numbers(
                            text):
                            extracted_fname = text
                            if extracted_fname is not None:
                                firstname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_fname)

                if firstname_value_block is not None:
                    # print("Extracted by firstname block")
                    return firstname_value_block

            # if first middle name not extracted, try using gender block upper
            gender_upper_block = None
            for val in result:
                text = val[1][0].lower()
                if gender_upper_block is None:
                    if check_gender_upper_block(text):
                        gender_upper_block = val
                        topmost_y_gender = max(x[1] for x in gender_upper_block[0])
                        lowermost_y_gender = y_lower(gender_upper_block[0])
                        leftmost_x_gender = min(x[0] for x in gender_upper_block[0])
                        rightmost_x_gender = max(x[0] for x in gender_upper_block[0])
                        break

            if gender_upper_block is not None:
                for val in result:
                    topmost_y_cur = max(x[1] for x in val[0])
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()

                    if diff is None:
                        if lowermost_y_cur < lowermost_y_gender and topmost_y_cur < topmost_y_gender and not has_numbers(
                                text):
                            extracted_name = val[1][0].lower()
                            if extracted_name is not None:
                                firstname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_gender)
                                diff_h = abs(leftmost_x_cur - leftmost_x_gender)

                    else:
                        if lowermost_y_cur < lowermost_y_gender and abs(
                                lowermost_y_cur - lowermost_y_gender) < diff and abs(
                            leftmost_x_cur - leftmost_x_gender) < 100 and not has_numbers(text):
                            extracted_name = val[1][0].lower()

                            if extracted_name is not None:
                                firstname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_gender)
                                diff_h = abs(leftmost_x_cur - leftmost_x_gender)

                if firstname_value_block is not None:
                    # print("Extracted by comparing gender upper block")
                    return firstname_value_block



        except Exception as e:
            # print(str(e))
            pass

        return None

    def process_first_middle_name(result):
#         try:
        if True:
            res = extract_firstname(result)

            if res is None:
                return DEFAULT_RES, DEFAULT_RES

            first_middle_name = res[-1][0]
            firstname_score= res[-1][1]
            middlename_score = res[-1][1]

            if first_middle_name is None:
                return DEFAULT_RES, DEFAULT_RES

            firstname = None
            middlename = None
            if ',' in first_middle_name:
                firstname = first_middle_name.split(',')[0]
                middlename = first_middle_name.split(',')[1]

            elif '.' in first_middle_name:
                firstname = first_middle_name.split('.')[0]
                middlename = first_middle_name.split('.')[1]

            else:
                firstname = first_middle_name
                # middlename = np.nan
                middlename = None
                middlename_score = 0.0

            return (firstname, firstname_score), (middlename, middlename_score)
#         except Exception as e:
#             # print(str(e))
#             return DEFAULT_RES, DEFAULT_RES

    def extract_gender(result):
        try:
            gender_value_block = None
            gender_block = None
            diff = None
            #         print("result: ", result[1])
            ## Extract using Surname
            for val in result:
                text = val[1][0].lower()
                #             print("text for matching lastname block: ", text)
                if gender_block is None:

                    if check_gender_text(text):
                        gender_block = val
                        lowermost_y_gender = y_lower(gender_block[0])
                        leftmost_x_gender = min(x[0] for x in gender_block[0])
                        rightmost_x_gender = max(x[0] for x in gender_block[0])
                        break
            if gender_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_gender and \
                                rightmost_x_cur > leftmost_x_gender and \
                                rightmost_x_gender > leftmost_x_cur:
                            extracted_gender = text
                            if extracted_gender is not None:
                                gender_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_gender)
                    else:
                        if lowermost_y_cur > lowermost_y_gender and abs(
                                lowermost_y_cur - lowermost_y_gender) < diff and \
                                rightmost_x_cur > leftmost_x_gender and \
                                rightmost_x_gender > leftmost_x_cur:
                            extracted_gender = text
                            if extracted_gender is not None:
                                gender_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_gender)

                if gender_value_block is not None:
                    # print("Extracted by gender block")
                    return gender_value_block

        except Exception as e:
            # print(str(e))
            pass

        return None

    def process_gender(result):
        try:
            res = extract_gender(result)
            if res is None:
                return DEFAULT_RES

            gender = process_extracted_gender(res[-1][0])
            gender_score = res[-1][1]

            # gender_num = None
            # if gender=='M':
            #     gender_num = '2'
            # elif gender=='F':
            #     gender_num = '1'

            return gender, gender_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    def extract_id_no(result):
        try:
            id_value_block = None
            id_block = None
            diff = None
            #         print("result: ", result[1])
            for val in result:
                text = val[1][0].lower()

                if id_block is None:

                    if check_id_no_text(text):
                        id_block = val
                        lowermost_y_id = y_lower(id_block[0])
                        leftmost_x_id = min(x[0] for x in id_block[0])
                        rightmost_x_id = max(x[0] for x in id_block[0])
                        break
            if id_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_id and \
                                rightmost_x_cur > leftmost_x_id and \
                                rightmost_x_id > leftmost_x_cur:
                            extracted_id_no = text
                            if extracted_id_no is not None:
                                id_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_id)
                    else:
                        if lowermost_y_cur > lowermost_y_id and abs(
                                lowermost_y_cur - lowermost_y_id) < diff and \
                                rightmost_x_cur > leftmost_x_id and \
                                rightmost_x_id > leftmost_x_cur:
                            extracted_id_no = text
                            if extracted_id_no is not None:
                                id_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_id)

                if id_value_block is not None:
                    # print("Extracted by id name block")
                    return id_value_block

            ## ID value block cannot be extracted, looking for nationality block
            nationality_block = None
            rep_of_mw_block = None
            for val in result:
                text = val[1][0].lower()

                if nationality_block is None:
                    if check_rep_of_mw_text(text):
                        rep_of_mw_block = val
                        lowermost_y_rep_mw = y_lower(rep_of_mw_block[0])
                        leftmost_x_rep_mw = min(x[0] for x in rep_of_mw_block[0])
                        rightmost_x_rep_mw = max(x[0] for x in rep_of_mw_block[0])

                    if match_MW1_text(text):
                        nationality_block = val
                        lowermost_y_nationality = y_lower(nationality_block[0])
                        leftmost_x_nationality = min(x[0] for x in nationality_block[0])
                        rightmost_x_nationality = max(x[0] for x in nationality_block[0])

                        if rightmost_x_nationality > leftmost_x_rep_mw and leftmost_x_nationality > leftmost_x_rep_mw:
                            break
                        else:
                            nationality_block = None
                            continue

            if nationality_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()

                    if diff is None:
                        if rightmost_x_cur < rightmost_x_nationality and leftmost_x_cur < leftmost_x_nationality:
                            id_no = val[1][0]
                            if id_no is not None:
                                id_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_nationality)

                    if diff is not None and abs(lowermost_y_cur - lowermost_y_nationality) < diff:
                        if rightmost_x_cur < rightmost_x_nationality and leftmost_x_cur < leftmost_x_nationality:
                            id_no = val[1][0]
                            if id_no is not None:
                                id_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_nationality)

                if id_value_block is not None:
                    # print("Extracted by Nationality")
                    return id_value_block

        except Exception as e:
            # print(str(e))
            pass

        return None

    def process_id_no(result):
        try:
            res = extract_id_no(result)

            if res is None:
                return DEFAULT_RES

            id_no = re.sub('[O]', '0', res[-1][0])
            id_no_score = res[-1][1]

            return id_no, id_no_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    def extract_lastname(result):
        try:
            lastname_value_block = None
            lastname_block = None
            diff = None
            #         print("result: ", result[1])
            ## Extract using Surname
            for val in result:
                text = val[1][0].lower()
                if lastname_block is None:

                    if check_lastname_text(text):

                        lastname_block = val
                        lowermost_y_lname = y_lower(lastname_block[0])
                        leftmost_x_lname = min(x[0] for x in lastname_block[0])
                        rightmost_x_lname = max(x[0] for x in lastname_block[0])
                        break
            if lastname_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()
                    if diff is None:
                        if lowermost_y_cur > lowermost_y_lname and \
                                rightmost_x_cur > leftmost_x_lname and \
                                rightmost_x_lname > leftmost_x_cur and not has_numbers(
                            text) and count_words(text) == 1:
                            extracted_lname = text
                            if extracted_lname is not None:
                                lastname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_lname)
                    else:
                        if lowermost_y_cur > lowermost_y_lname and abs(
                                lowermost_y_cur - lowermost_y_lname) < diff and \
                                rightmost_x_cur > leftmost_x_lname and \
                                rightmost_x_lname > leftmost_x_cur and not has_numbers(
                            text) and count_words(text) == 1:
                            extracted_lname = text
                            if extracted_lname is not None:
                                lastname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_lname)

                if lastname_value_block is not None:
                    # print("Extracted by lastname block")
                    return lastname_value_block

            ## Lastname cannot be extracted, extract using Firstname
            firstname_block = None
            for val in result:
                text = val[1][0].lower()
                if firstname_block is None:
                    if check_firstname_text(text):
                        firstname_block = val
                        lowermost_y_fname = y_lower(firstname_block[0])
                        leftmost_x_fname = min(x[0] for x in firstname_block[0])
                        rightmost_x_fname = max(x[0] for x in firstname_block[0])
                        break

            if firstname_block is not None:
                for val in result:
                    lowermost_y_cur = y_lower(val[0])
                    rightmost_x_cur = max(x[0] for x in val[0])
                    leftmost_x_cur = min(x[0] for x in val[0])
                    text = val[1][0].lower()

                    if diff is None:
                        if lowermost_y_cur < lowermost_y_fname and not has_numbers(text) and count_words(text) == 1:
                            extracted_lname = text
                            if extracted_lname is not None:
                                lastname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_fname)
                    else:
                        if lowermost_y_cur < lowermost_y_fname and abs(
                                lowermost_y_cur - lowermost_y_fname) < diff and not has_numbers(text) and count_words(
                            text) == 1:
                            extracted_lname = text
                            if extracted_lname is not None:
                                lastname_value_block = val
                                diff = abs(lowermost_y_cur - lowermost_y_fname)

                if lastname_value_block is not None:
                    # print("Extracted by comparing firstname block")
                    return lastname_value_block


        except Exception as e:
            # print(str(e))
            pass

        return None

    def process_lastname(result):
        try:
            res = extract_lastname(result)
            if res is None:
                return DEFAULT_RES

            lastname = check_lastname(res[-1][0])
            lastname_score = res[-1][1]

            return lastname, lastname_score
        except Exception as e:
            # print(str(e))
            return DEFAULT_RES

    # Extracting all fields

    doe_res, doe_score = process_date_of_expiry(ocr_detections)
    if doe_res is not None:
        doe_res = doe_res.strftime('%d-%b-%Y')

    doi_res, doi_score = process_date_of_issue(ocr_detections)
    if doi_res is not None:
        doi_res = doi_res.strftime('%d-%b-%Y')

    dob_res, dob_score = process_date_of_birth(ocr_detections)
    if dob_res is not None:
        dob_res = dob_res.strftime('%d-%b-%Y')

    (firstname, firstname_score), (middlename, middlename_score) = process_first_middle_name(ocr_detections)

    gender, gender_score =process_gender(ocr_detections)

    id_no, id_no_score = process_id_no(ocr_detections)

    lastname, lastname_score = process_lastname(ocr_detections)

    demographic_fields = [
        { "name": OCRFieldNames.FIRST_NAME, "value": firstname,"coordinates": None,"score": firstname_score},
        { "name": OCRFieldNames.LAST_NAME, "value": lastname,"coordinates": None,"score": lastname_score},
        { "name": OCRFieldNames.MIDDLE_NAME, "value": middlename,"coordinates": None,"score": middlename_score},
        { "name": OCRFieldNames.GENDER, "value": gender,"coordinates": None,"score": gender_score},
        { "name": OCRFieldNames.DATE_OF_BIRTH, "value": dob_res,"coordinates": None,"score": dob_score},
        { "name": OCRFieldNames.ID_NUMBER, "value": id_no,"coordinates": None,"score": id_no_score},
        { "name": OCRFieldNames.DATE_OF_ISSUE, "value": doi_res,"coordinates": None,"score": doi_score},
        { "name": OCRFieldNames.DATE_OF_EXPIRY, "value": doe_res,"coordinates": None,"score": doe_score}
    ] 

    return demographic_fields


