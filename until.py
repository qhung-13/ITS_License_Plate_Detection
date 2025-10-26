import string
import easyocr
import re
import cv2

VIETNAM_PLATE_ALLOW_LIST = 'ABCDEFGHIJKLMNPSTUVXYZ0123456789'

# Initialize the OCR reader
reader = easyocr.Reader(['en', 'vi'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def write_csv(results, filename):
    with open(filename, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_box',
                                                'license_plate_bbox', 'license_plate_bbox_score',
                                                'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                    'license_plate' in results[frame_nmr][car_id].keys() and \
                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
    f.close()

def license_complies_format(text):
    text_orig = text.upper().replace(' ', '')
    # text = text.replace('O', '0').replace('I', '1')
    text_clean = text.replace('.', '').replace('-', '')

    patterns_clean = [
        r'^[0-9]{2}[A-Z][0-9]{4,5}$',  # 30A12345
        r'^[A-Z]{2}[0-9]{5}$',  # TM12345
        r'^(NN|NG)[0-9]{3,5}$'  # NN12345
    ]

    for p_clean in patterns_clean:
        if re.match(p_clean, text_clean):
            return True

    patterns_original = [
        r'^[0-9]{2}[A-Z]-?[0-9]{4,5}$',
        r'^[0-9]{2}[A-Z]-?[0-9]{3}\.[0-9]{2}$',
        r'^[A-Z]{2}-?[0-9]{5}$',
        r'^(NN|NG)-?[0-9]{3,5}$'
    ]

    for p_orig in patterns_original:
        if re.match(p_orig, text_orig):
            return True

    return False
    # text_no_special_chars = text.replace('.', '').replace('-', '')
    #
    # for p in patterns:
    #     clean_text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    #
    #     patterns_clean = [
    #         r'^[0-9]{2}[A-Z][0-9]{4,5}$',  # 30A12345
    #         r'^[0-9]{2}[A-Z][0-9]{3}[0-9]{2}$',  # 30A12345 (cho trường hợp 123.45)
    #         r'^[A-Z]{2}[0-9]{5}$',  # TM12345
    #         r'^(NN|NG)[0-9]{3,5}$'  # NN12345
    #     ]
    #     for p_clean in patterns_clean:
    #         if re.match(p_clean, clean_text):
    #             return True
    #
    #     if re.match(p, text):
    #         return True
    # return False


def format_license(text):
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

    replacements = {
        'O': '0', 'I': '1', 'L': '1', 'Z': '2',
        'S': '5', 'B': '8', 'G': '6', 'Q': '9'
    }

    part_head = ""
    part_tail = ""

    match_private = re.search(r'^([0-9]{2}[A-Z])([0-9]+)$', text)
    match_other = re.search(r'^([A-Z]{2}|NN|NG)([0-9]+)$', text)

    if match_private:
        part_head = match_private.group(1)
        part_tail = match_private.group(2)
    elif match_other:
        part_head = match_other.group(1)
        part_tail = match_other.group(2)
    else:
        part_tail = text

    for old, new in replacements.items():
        part_tail = part_tail.replace(old, new)

    return part_head + part_tail

    # for old, new in replacements.items():
    #     text = text.replace(old, new)
    # text = text.replace('.', '').replace('-', '').replace(' ', '')
    # return text

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, card_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_idx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_idx]

    return -1, -1, -1, -1, -1

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop,
                                 allowlist=VIETNAM_PLATE_ALLOW_LIST,
                                 paragraph=True,
                                 detail=0)

    print("OCR detections raw:", detections)
    cv2.imshow("OCR_input", license_plate_crop)
    cv2.waitKey(1)

    if not detections:
        return None, None

    text = detections[0].upper().replace(' ', '')
    print(f"OCR Raw Text: {text}")

    if license_complies_format(text):
        print(f"Text '{text}' PASSED format check.")
        final_text = format_license(text)
        print(f"Final Text: {final_text}")
        return final_text, 1.0
    else:
        print(f"Text '{text}' FAILED format check.")
        return None, None
    # for detection in detections:
    #     bbox, text, score = detection
    #     print(f"OCR: '{text}' ({score:.2f})")
    #
    #     text = text.upper().replace(' ', '')
    #
    #     if license_complies_format(text):
    #         return format_license(text), score
