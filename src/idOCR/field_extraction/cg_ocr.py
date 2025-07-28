import fuzzysearch
import re
from difflib import SequenceMatcher
import cv2
import re
import logging

logger = logging.getLogger(__name__)
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
            raw_result = "National Identity Card"
        elif any(merchant_id_matches):
            # Check for override keywords within merchant matches
            if any(fuzzysearch.find_near_matches('passeport', text, max_l_dist=1)):
                raw_result = "Others"
            elif any(fuzzysearch.find_near_matches('permisdeconduire', text, max_l_dist=1)):
                raw_result = "Driving License"
            elif any([fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in
                      ['cartedelecteur', 'cartedidetiteconsulaire', 'cartedidentitepourrefugie']]):
                raw_result = "Others"
            else:
                raw_result = "Merchant Card"
        elif any(student_id_matches):
            # Check for override keywords within student matches
            if any(fuzzysearch.find_near_matches('passeport', text, max_l_dist=1)):
                raw_result = "Others"
            elif any(fuzzysearch.find_near_matches('permisdeconduire', text, max_l_dist=1)):
                raw_result = "Driving License"
            elif any([fuzzysearch.find_near_matches(keyword, text, max_l_dist=1) for keyword in
                      ['cartedelecteur', 'cartedidetiteconsulaire', 'cartedidentitepourrefugie']]):
                raw_result = "Others"
            else:
                raw_result = "Others"
        elif any(dl_matches):
            raw_result = "Driving License"
        elif any(passport_matches):
            raw_result = "Others"
        else:
            raw_result = "Others"

        # Map to standardized labels
#         standardized_label = OCR_LABEL_MAPPING.get(raw_result)

        logger.debug(f"OCR detection: raw='{raw_result}' ")
        return raw_result

    except Exception as e:
        logger.error(f"Error in OCR ID type detection: {str(e)}")
        return None

# Case 1
### Relative distance logic for all the fields
def remove_special_characters(string):
    modified_string = ''
    for item in string.lower():
        if(item in '0123456789abcdefghijklmnopqrstuvwxyz'):
            modified_string+=item
    return modified_string

def date_check_1(string):
    string=string.lower().strip()
    flag=False
    if (SequenceMatcher(None, "lieudenaissance", string).ratio()>=0.6):
        flag=True
    return flag

def extract_based_on_relative_distance(updated_results,key_ratio,idtype):
    republic_cent_y=updated_results[0]['cent_y']
    id_card_cent_y=updated_results[1]['cent_y']
#     print(republic_cent_y,id_card_cent_y)
    if idtype=='National Identity Card':
        for item in updated_results:
            if(len(fuzzysearch.find_near_matches('republiqueducongo', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                republic_cent_y = item['cent_y']
            if(len(fuzzysearch.find_near_matches('cartenationaledidentite', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                id_card_cent_y = item['cent_y']
    if idtype=='Merchant Card':
        for item in updated_results:
            if(len(fuzzysearch.find_near_matches('republiqueducongo', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                republic_cent_y = item['cent_y']
            if(len(fuzzysearch.find_near_matches('unitetravailprogres', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                id_card_cent_y = item['cent_y']
    if idtype=='Driving License':
        for item in updated_results:
            if(len(fuzzysearch.find_near_matches('republiqueducongo', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                republic_cent_y = item['cent_y']
            if(len(fuzzysearch.find_near_matches('permisdeconduire', item['text'][0].replace(' ','').lower(), max_l_dist=4))>=1):
                id_card_cent_y = item['cent_y']
    
    predicted_distance = key_ratio*(id_card_cent_y-republic_cent_y)+republic_cent_y

    distance = list()
    text_sorted_with_distance = list()
    for item in updated_results:
        distance.append(abs(item['cent_y']-predicted_distance))
        text_sorted_with_distance.append(item['text'][0])


    return text_sorted_with_distance[distance.index(min(distance))]

### adding score and box for values
# back_prediction = list()
img_width = 0
img_height = 0

def extract_kyc_fields(results ,back_prediction, image):
    idtype=get_id_type_by_ocr(results)
    if idtype!='National Identity Card' and idtype!='Merchant Card' and idtype!='Driving License':
        first_name_dict = {"name": OCRFieldNames.FIRST_NAME,
                       "value": None,
                       "coordinate":None,
                       "score":None}
    
        last_name_dict = {"name": OCRFieldNames.LAST_NAME,
                           "value":None,
                           "coordinate":None,
                           "score":None}

        date_of_birth_dict = {"name": OCRFieldNames.DATE_OF_BIRTH ,
                              "value":None,
                              "coordinate":None,
                              "score":None}

        gender_dict = {"name": OCRFieldNames.GENDER,
                       "value":None,
                       "coordinate":None,
                       "score":None}

        place_of_birth_dict = {"name": OCRFieldNames.PLACE_OF_BIRTH,
                               "value":None,
                               "coordinate":None,
                               "score":None}

        id_number_dict = {"name": OCRFieldNames.ID_NUMBER,
                          "value":None,
                          "coordinate":None,
                          "score":None}
        doe_dict = {"name": OCRFieldNames.DATE_OF_EXPIRY,
                      "value":None,
                      "coordinate":None,
                      "score":None} 
        result = [first_name_dict, last_name_dict , date_of_birth_dict, gender_dict, place_of_birth_dict, id_number_dict, doe_dict]

        return result
    ## Doing OCR and getting text and their co-ordinates from image
    boxes = [line[0] for line in results]
    txts = [line[1][0] for line in results]
    scores = [line[1][1] for line in results]
    
    ## Sorting texts with respect to y height
    updated_results = []
    for coords, text in results:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        x4, y4 = coords[3]

        cent_y = (y1 + y2 + y3 + y4) / 4
        cent_x = (x1 + x2 + x3 + x4) / 4

        updated_results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })

    updated_results = sorted(updated_results, key=lambda x: x['cent_y'])
    
    
    txt_list = list()
    cent_y_list = list()
    score_list = list()
    bbox_list = list()
    for item in updated_results:
        txt_list.append(item['text'][0])
        cent_y_list.append(item['cent_y'])
        score_list.append(item['text'][1])
        bbox_list.append(item['coords'])
    
    
    score_result = {
                    'First Name':None,
                    'Last Name':None,
                    'Date of Birth':None,
                    'Gender':None,
                    'Place of Birth':None,
                    'ID Number':None,
                    'ID Type':None
                }

    bbox_result = {
                    'First Name':None,
                    'Last Name':None,
                    'Date of Birth':None,
                    'Gender':None,
                    'Place of Birth':None,
                    'ID Number':None,
                    'ID Type':None
                }
    
    
    nom_idx = None
    prenoms_idx = None
    dob_idx = None
    gender_idx = None
    pob_idx = None
    id_number_idx = None
    for i,item in enumerate(txt_list):
        if(len(fuzzysearch.find_near_matches('nom', item.replace(' ','').lower(), max_l_dist=1))>=1):
            if(not nom_idx):
                nom_idx = i
        if (nom_idx is not None) and (idtype=='Merchant Card' or idtype=='Driving License'):
            id_number_idx=nom_idx

        if(len(fuzzysearch.find_near_matches('prenoms', item.replace(' ','').lower(), max_l_dist=2))>=1):
            if(not prenoms_idx):
                prenoms_idx = i

        if(len(fuzzysearch.find_near_matches('datedenaissance', item.replace(' ','').lower(), max_l_dist=4))>=1):
            if(not dob_idx):
                dob_idx = i

        if(len(fuzzysearch.find_near_matches('sexe', item.replace(' ','').lower(), max_l_dist=1))>=1):
            if(not gender_idx):
                gender_idx = i

        if(len(fuzzysearch.find_near_matches('lieudenaissance', item.replace(' ','').lower(), max_l_dist=2))>=1) or (date_check_1(item)):
                pob_idx = i
        if idtype=='National Identity Card':
            if(len(fuzzysearch.find_near_matches('cnin', item.replace(' ','').lower(), max_l_dist=1))>=1):
                if(not id_number_idx):
                    id_number_idx = i

    
    ### Writing Logic for last name
    try:
        last_name = txt_list[nom_idx+1]
    except:
        try:
            last_name = txt_list[prenoms_idx-1]
        except:
            last_name = None
            
    if(1):
        try:
            if idtype=='National Identity Card':
                last_name = extract_based_on_relative_distance(updated_results,2.3,idtype)
            if idtype=='Merchant Card':
                last_name = extract_based_on_relative_distance(updated_results,5,idtype)
            if idtype=='Driving License':
                last_name = extract_based_on_relative_distance(updated_results,2,idtype)
        except:
            pass
            
    ### Writing Logic for first name
    try:
        first_name = txt_list[prenoms_idx+1]
    except:
        try:
            first_name = txt_list[dob_idx-1]
        except:
            first_name = None
            
    if(1):
        try:
            if idtype=='National Identity Card':
                first_name = extract_based_on_relative_distance(updated_results,3.15,idtype)
            if idtype=='Merchant Card':
                first_name = extract_based_on_relative_distance(updated_results,6.5,idtype)
            if idtype=='Driving License':
                first_name = extract_based_on_relative_distance(updated_results,5.5,idtype)
        except:
            pass
        
    if(first_name==last_name):
        try:
            if idtype=='National Identity Card':
                last_name = extract_based_on_relative_distance(updated_results,2.3,idtype)
            if idtype=='Merchant Card':
                last_name = extract_based_on_relative_distance(updated_results,5,idtype)
            if idtype=='Driving License':
                last_name = extract_based_on_relative_distance(updated_results,2,idtype)
        except:
            pass
        
    ### Writing DOB logic        
    try:
        dob = txt_list[dob_idx+1]
        dob=dob.replace('.','').replace(' ','').replace('/','')
#         print(txt_list,"dob_idx",txt_list[dob_idx+1])
        try:
            int(dob)
        except:
            dob=None
        else:
            if dob.isdigit():
                if idtype!='Merchant Card':
                    dob = "".join(reversed(dob))
                    year = "".join(reversed(dob[:4]))
                    month = "".join(reversed(dob[4:6]))
                    day = "".join(reversed(dob[6:]))
#                     print("Hi",year,month,day,dob)
                    dob = year + "-" + month + "-" + day
                else:
    #                 dob = "".join(reversed(dob))
                    year="20"+dob[-2:]
                    if int(dob[-2:])>25:
                        year = "19"+dob[-2:]
                    month = "".join(dob[2:4])
                    day = "".join(dob[0:2])
    #                 print("Hi",year,month,day,dob)
                    dob = year + "-" + month + "-" + day
                score_result['Date of Birth'] = score_list[i]
                bbox_result['Date of Birth'] = bbox_list[i]
            
    except:
        dob = None
        
    if(dob==None):
        try:
#             print("I had to enter")
            ### Logic improvemnet for DOB
            ocr_text = ' '.join(txts).replace('.','').replace(' ','').replace('/','')
#             print("OCR:",ocr_text)
            pattern = "\d{2}\d{2}\d{4}"
            if idtype=='Merchant Card':
                pattern = "\d{2}\d{2}\d{2}"
            dates = re.findall(pattern, ocr_text)
#             print("Hilo",dates)
            dob=dates[0]
            dob = remove_special_characters(dob)
            if idtype!='Merchant Card':
                dob = "".join(reversed(dob))
                year = "".join(reversed(dob[:4]))
                month = "".join(reversed(dob[4:6]))
                day = "".join(reversed(dob[6:]))
#                 print("Hemlo",year,month,day,dob)
                dob = year + "-" + month + "-" + day
            else:
#                 dob = "".join(reversed(dob))
                year="20"+dob[-2:]
                if int(dob[-2:])>25:
                    year = "19"+dob[-2:]
                month = "".join(dob[2:4])
                day = "".join(dob[0:2])
                dob = year + "-" + month + "-" + day
#                 year = "20" + doe[-2:]
#             month = doe[2:4]
#             day = doe[0:2]
#                 print("Hemlo2",year,month,day,dob)
            score_result['Date of Birth'] = score_list[i]
            bbox_result['Date of Birth'] = bbox_list[i]
        except:
            pass
#     print("*****",dob)
    # if dob is not None:
        
        
    ## Writing logic for gender    
    try:
        
        if(('m' in txt_list[gender_idx].lower().replace(' ','')) or ('w' in txt_list[gender_idx].lower().replace(' ',''))):
            gender='M'
        else:
            gender='F'
        score_result['Gender'] = score_list[gender_idx]
        bbox_result['Gender'] = bbox_list[gender_idx]
        
    except:
        gender=None
    
    if(gender_idx!=None):
        if(len(txt_list[gender_idx].lower().replace(' ',''))>5): gender=None
    
    if(gender==None):
        try:
            if idtype=='National Identity Card':
                gender_text = extract_based_on_relative_distance(updated_results,5.54,idtype)
            if idtype=='Merchant Card':
                gender_text = extract_based_on_relative_distance(updated_results,10.5,idtype)
            
            #print(f'second logic {gender_text}')
            if(('m' in gender_text.lower().replace(' ','')) or ('w' in gender_text.lower().replace(' ',''))):
                gender='M'
            else:
                gender='F'
            for i,item in enumerate(txt_list):
                if(gender_text.lower().replace(' ','') in item.lower().replace(' ','')):
                    score_result['Gender'] = score_list[i]
                    bbox_result['Gender'] = bbox_list[i]
        except:
            pass
    if(gender_idx!=None):
        pass
        #print(f'first logic {txt_list[gender_idx]}')
    #print('******')
        
    try:
        pob = ''.join(txt_list[pob_idx+1])
    except:
        pob=None
        
    if(pob==None):
        try:
            if idtype=='National Identity Card':
                pob = extract_based_on_relative_distance(updated_results,5.1,idtype)
            if idtype=='Merchant Card':
                pob = extract_based_on_relative_distance(updated_results,9,idtype)
            if idtype=='Driving License':
                pob = extract_based_on_relative_distance(updated_results,5,idtype)
            pob=re.sub(r'[^a-zA-Z\s]', '', pob)
        except:
            pob = None
        
        
    id_number=None
    try:
        if idtype=='National Identity Card':
            id_number_token = list()
            k=''
#             print('txt_l:',txt_list[-1])
            for token in txt_list[-1].split('-'):
                id_number_token+=token.split(' ')
#             print('id_token',id_number_token)
            if len(id_number_token[-1])==2:
                    k=id_number_token[-1]
            if len(id_number_token)==3:
                if len(id_number_token[-1])==2:
                    k=id_number_token[-1]
            if len(id_number_token)==2:
                if len(id_number_token[-1])==2:
                    k=id_number_token[-1]
            for token in id_number_token:
                if(len(token)>10):
#                     print("Hi i am on",token,token[0],token[len(token)-13:len(token)][0])
                    try:
                        int(token)
                    except:
#                         print('I did come here')
                        try:
                            int(token[len(token)-13:len(token)][0])
                        except:
#                             print("Hi i am on here")
                            id_number=token[len(token)-13:len(token)]+k
#                         print("Hi, I have reached here my id_number is:",id_number)
        if idtype=='Merchant Card' or idtype=='Driving License':
            id_number=None
        if idtype=='Driving License':
            tokens = list()
            for token in ' '.join(txts).split(' '):
                tokens+=token.split(':')
            
            have_charcater = False
            have_digit = False
            for token in reversed(tokens):
                have_charcater = False
                have_digit = False
                tmp_token=''
                for character in token:
                    if(character in 'abcdefghijklmnopqrstuvwxyz'):
                        pass
                    else:
                        tmp_token+=character
                token = tmp_token
                        
                if(re.match('^[A-Z0-9]*$',token) and len(token)>=11):
                    for character in token:
                        if(character in '0123456789'):
                            have_digit=True
                        if(character in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                            have_charcater=True
#                     print("Token:",token,"has_digit",have_digit,"has_char:",have_charcater)
#                     if(have_digit and not have_charcater):
#                         print("Hi,i reached here for k,",token)
#                         if len(token)==2:
#                             k=token
                    if(have_digit and have_charcater):
#                             print("I am on this token",token,token[len(token)-11:len(token)][0])
                            try:
                                int(token[len(token)-11:len(token)][0])
                            except:
                                id_number=token[len(token)-11:len(token)]
#                                 print("Hello,my id number:",id_number)
                                break

    except:
        pass
    try:
        if id_number==None:
            if idtype=='National Identity Card':
                id_number_token = list()
                k=''
                for token in txt_list[id_number_idx].split('-'):
                    id_number_token+=token.split(' ')
                if len(id_number_token[-1])==2:
                        k=id_number_token[-1]
                if len(id_number_token)==3:
                    if len(id_number_token[-1])==2:
                        k=id_number_token[-1]
                for token in id_number_token:
                    if(len(token)>10):
                        try:
                            int(token)
                        except:
                            try:
                                int(token[len(token)-13:len(token)][0])
                            except:
                                id_number=token[len(token)-13:len(token)]+k
            if idtype=='Merchant Card' or idtype=='Driving License':
                id_number=None

    except:
        pass
    
    
    ## Second method if id_number = None
    try:
        if idtype=='National Identity Card':
            if(id_number==None):
                id_number_token = list()
                k=''
                for token in txt_list[id_number_idx-1].split('-'):
                    id_number_token+=token.split(' ')
                if len(id_number_token[-1])==2:
                        k=id_number_token[-1]
#                 if len(id_number_token)==3:
#                     if len(id_number_token[-1])==2:
#                         k=id_number_token[-1]

                for token in id_number_token:
                    if(len(token)>10):
                        try:
                            int(token)
                        except:
#                             print(int(token[len(token)-13:len(token)])[0])
                            try:
                                int(token[len(token)-13:len(token)][0])
                            except:
                                id_number=token[len(token)-13:len(token)]+k
#                             print("Hi,my id number:",id_number,k)
    except:
        pass
    
    if(id_number==None):
        if idtype=='National Identity Card':
            tokens = list()
            k=''
            for token in ' '.join(txts).split(' '):
                tokens+=token.split('-')
            if len(tokens[-1])==2:
                    k=tokens[-1]
#             print("Hi my tokens are: ",tokens,"and k is ",k)
#             if len(tokens)==3:
#                 if len(tokens[-1])==2:
#                     k=tokens[-1]
            have_charcater = False
            have_digit = False
            for token in reversed(tokens):
                have_charcater = False
                have_digit = False
                tmp_token=''
                for character in token:
                    if(character in 'abcdefghijklmnopqrstuvwxyz'):
                        pass
                    else:
                        tmp_token+=character
                token = tmp_token
                if len(token)==2:
#                     print("hi, i came here",token)
                    for character in token:
                        if(character in '0123456789'):
                                have_digit=True
                        else:
                            have_digit=False
                            break
                    if(have_digit):
#                         print("i made it here")
                        k=token
#                         print("i really made it here",k)
                        
                if(re.match('^[A-Z0-9]*$',token) and len(token)>=13):
                    for character in token:
                        if(character in '0123456789'):
                            have_digit=True
                        if(character in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                            have_charcater=True
#                     print("Token:",token,"has_digit",have_digit,"has_char:",have_charcater)
#                     if(have_digit and not have_charcater):
#                         print("Hi,i reached here for k,",token)
#                         if len(token)==2:
#                             k=token
                    if(have_digit and have_charcater):
#                             print("I am on this token",token,token[len(token)-13:len(token)][0])
                            try:
                                int(token[len(token)-13:len(token)][0])
                            except:
                                id_number=token[len(token)-13:len(token)]+k
#                                 print("Hello,my id number:",id_number)
                                break
#             if id_number is not None:
#                     id_number=id_number+k
        if idtype=='Merchant Card':
            id_number=extract_based_on_relative_distance(updated_results,3,idtype)
        if idtype=='Driving License':
            id_number=extract_based_on_relative_distance(updated_results,6.5,idtype)[-11:]

    for i,item in enumerate(txt_list):
        if first_name is not None:
            if(first_name.lower().replace(' ','') in item.lower().replace(' ','')):
                score_result['First Name'] = score_list[i]
                bbox_result['First Name'] = bbox_list[i]
        else :
            score_result['First Name'] = None
            bbox_result['First Name'] = None
        if last_name is not None:
            if(last_name.lower().replace(' ','') in item.lower().replace(' ','')):
                score_result['Last Name'] = score_list[i]
                bbox_result['Last Name'] = bbox_list[i]
        else:
            score_result['Last Name'] = None
            bbox_result['Last Name'] = None

        if dob is not None:
#             print("Hi i am here",dob)
            if(dob.lower().replace(' ','') in item.lower().replace(' ','')):
                dob = remove_special_characters(dob)
                dob = "".join(reversed(dob))
                year = "".join(reversed(dob[:4]))
                month = "".join(reversed(dob[4:6]))
                day = "".join(reversed(dob[6:]))
                dob = year + "-" + month + "-" + day
                score_result['Date of Birth'] = score_list[i]
                bbox_result['Date of Birth'] = bbox_list[i]
        else:
            score_result['Date of Birth'] = None
            bbox_result['Date of Birth'] = None
 
        if pob is not None:
            if(pob.lower().replace(' ','') in item.lower().replace(' ','')):
                score_result['Place of Birth'] = score_list[i]
                bbox_result['Place of Birth'] = bbox_list[i]
        else:
            score_result['Place of Birth'] = None
            bbox_result['Place of Birth'] = None
       
        if id_number is not None:
            if(id_number.lower().replace(' ','') in item.lower().replace(' ','').replace('-','')):
                score_result['ID Number'] = score_list[i]
                bbox_result['ID Number'] = bbox_list[i]
        else:
            score_result['ID Number'] = None
            bbox_result['ID Number'] = None
    
    first_name_dict = {"name": OCRFieldNames.FIRST_NAME,
                       "value":first_name,
                       "coordinate":bbox_result['First Name'],
                       "score":score_result['First Name']}
    
    last_name_dict = {"name": OCRFieldNames.LAST_NAME,
                       "value":last_name,
                       "coordinate":bbox_result['Last Name'],
                       "score":score_result['Last Name']}
    
    date_of_birth_dict = {"name": OCRFieldNames.DATE_OF_BIRTH ,
                          "value":dob,
                          "coordinate":bbox_result['Date of Birth'],
                          "score":score_result['Date of Birth']}
    
    gender_dict = {"name": OCRFieldNames.GENDER,
                   "value":gender,
                   "coordinate":bbox_result['Gender'],
                   "score":score_result['Gender']}
    
    place_of_birth_dict = {"name": OCRFieldNames.PLACE_OF_BIRTH,
                           "value":pob,
                           "coordinate":bbox_result['Place of Birth'],
                           "score":score_result['Place of Birth']}
    
    id_number_dict = {"name": OCRFieldNames.ID_NUMBER,
                      "value":id_number,
                      "coordinate":bbox_result['ID Number'],
                      "score":score_result['ID Number']}
    
    doe_dict = date_of_expiry(back_prediction,idtype)
    
    result = [first_name_dict, last_name_dict , date_of_birth_dict, gender_dict, place_of_birth_dict, id_number_dict, doe_dict]

    return result

def date_of_expiry(results,idtype):
    if results is None:
        doe_dict = {"name": OCRFieldNames.DATE_OF_EXPIRY,
                      "value":None,
                      "coordinate":None,
                      "score":None} 
        return doe_dict
    
    pattern = r'\d{2}(?:[-./:]?)\d{2}(?:[-./:]?)\d{4}'
    if idtype=='Merchant Card':
        pattern=r'\d{2}(?:[-./:]?)\d{2}(?:[-./:]?)\d{2}'
    boxes = [line[0] for line in results]
    txts = [line[1][0] for line in results]
    scores = [line[1][1] for line in results]

    score_result = None

    bbox_result = None
    
    ## Sorting texts with respect to y height
    updated_results = []
    #     for coords, text in results:
    for coords, text in results:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        x4, y4 = coords[3]
    
        cent_y = (y1 + y2 + y3 + y4) / 4
        cent_x = (x1 + x2 + x3 + x4) / 4
    
        updated_results.append({
            'cent_x': cent_x,
            'cent_y': cent_y,
            'coords': coords,
            'text': text
        })
    
    updated_results = sorted(updated_results, key=lambda x: x['cent_y'])
    i = len(updated_results) - 1
    doe = None
    while i>=0:
        text = txts[i].lower().replace(" ","")
        match = re.search(pattern, text)
        if match is not None and len(text)<=12:
            doe = text
            doe = doe.replace(".","").replace("/","")
            year = "20" + doe[-2:]
            month = doe[2:4]
            day = doe[0:2]
            if int(day) > 31 or int(month)>12:
                doe = None
            else:
                doe = day + "." + month + "." + year
        if doe is not None:
            doe = remove_special_characters(doe)
            doe = "".join(reversed(doe))
            year = "".join(reversed(doe[:4]))
            month = "".join(reversed(doe[4:6]))
            day = "".join(reversed(doe[6:]))
            doe = year + "-" + month + "-" + day
            bbox_result = boxes[i]
            score_result = scores[i]
            break
        i = i-1

    doe_dict = {"name": OCRFieldNames.DATE_OF_EXPIRY,
                      "value":doe,
                      "coordinate":bbox_result,
                      "score":score_result} 
     
    return doe_dict