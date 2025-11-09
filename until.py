import numpy as np
import re
import cv2
import pytesseract
import  subprocess
import os

# --- Cấu hình Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Bảng ký tự cho phép ---
VIETNAM_PLATE_ALLOW_LIST = 'ABCDEFGHIJKLMNPSTUVXYZ0123456789'

# --- Chuyển đổi ký tự dễ nhầm ---
CHAR_FIX_MAP = {
    'O': '0', 'D': '0',
    'I': '1', 'L': '1',
    'S': '5',
    'B': '8',
    'G': '6',
    'Q': '9',
    'A': '4',
    'Z': '2',
    'E': '8',
    'T': '7'
}

LETTER_FIX_MAP = {
    '0': 'O',
    '6': 'G',
    '1': 'I',
    '5': 'S',
    '2': 'Z',
    '8': 'B',
    '4': 'A'
}

LETTER_TO_LETTER_FIX_MAP = {
    'O': 'A',
    'D': 'A',
    'Q': 'O'
}

def fix_plate_chars_smart(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 6 or len(text) > 9:
        return text

    if len(text) >= 3 and text[:2].isalpha():
        first_two_fixed = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text[:2])
        # nếu sau khi đổi có dạng 2 số đầu + 1 chữ => biển hợp lý
        if first_two_fixed.isdigit() and len(text) >= 3:
            text = first_two_fixed + text[2:]

    if text[0].isdigit() and text[1].isdigit():
        head = text[:2]
        letter = text[2] if len(text) > 2 else ''
        tail = text[3:] if len(text) > 3 else ''

        # Nếu ký tự thứ 3 là số thì OCR sai — sửa thành chữ gần đúng
        if letter.isdigit():
            letter = LETTER_FIX_MAP.get(letter, 'A')

        # Nếu ký tự thứ 3 là chữ, sửa các lỗi phổ biến
        else:
            letter = LETTER_FIX_MAP.get(letter, letter)
            letter = LETTER_TO_LETTER_FIX_MAP.get(letter, letter)

        # Fix phần số đằng sau
        tail_fixed = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in tail)

        return head + letter + tail_fixed

        # Biển kiểu đặc biệt (NN12345, NG12345, TM12345, v.v.)
    elif text[:2] in ['NN', 'NG', 'TM']:
        prefix = text[:2]
        tail = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text[2:])
        return prefix + tail

    return text

def fix_top_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())

    if(len(text) < 2):
        return text

    CHAR_FIX_MAP = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'S': '5',
        'B': '8',
        'G': '6',
        'Z': '2',
        'E': '8',
        'T': '7',
        'R': '4',
    }

    LETTER_FIX_MAP = {
        '0': 'O',
        '6': 'G',
        '1': 'I',
        '5': 'S',
        '2': 'Z',
        '8': 'B',
        '4': 'A'
    }

    PROVINCE_FIX_MAP = {
        'O': '8',  # O thường bị OCR đọc nhầm thay vì 8
        'D': '0',
        'Q': '0',
        'E': '4',
        'Z': '2',
        'S': '5',
        'T': '7',
        'B': '8',
    }

    CHAR_NOT_IN_PLATE = ['I', 'J', 'O', 'Q', 'R', 'W']

    FIX_THIRD_CHAR = {
        'R': 'A'
    }

    #----- 2 ki tu dau
    fixed_first_two = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text[:2])
    print(f"[OCR Fixed First] {fixed_first_two}")
    if (fixed_first_two[0] == '0'):
        fixed_first_two = ''.join(PROVINCE_FIX_MAP.get(ch, ch) for ch in text[:2])
        print(f"[OCR Fixed Second] {fixed_first_two}")
    if (not fixed_first_two.isdigit()):
        fixed_first_two = ''.join(PROVINCE_FIX_MAP.get(ch, ch) for ch in text[:2])
        print(f"[OCR Fixed Third] {fixed_first_two}")

    #---- ki tu thu 3
    third = ''
    rest = ''

    if len(text) > 2:
        third = text[2]
        if third.isdigit():
            third = LETTER_FIX_MAP.get(third, 'A')
        else:
            third = LETTER_FIX_MAP.get(third, third)
            if (third in CHAR_NOT_IN_PLATE):
                third = FIX_THIRD_CHAR.get(third, third)

        rest = text[3:]

    rest_fixed = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in rest)

    return (fixed_first_two + third + rest_fixed).strip()

def format_license_smart(text):
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

    if len(text) >= 7:
        head = text[:2]
        letter = text[2]
        tail = text[3:]
        # Trả về text đã được sửa bởi fix_plate_chars_smart
        return head + letter + tail
    return text
# =====================================================================
# Ghi kết quả ra file CSV
# =====================================================================
def write_csv(results, filename):
    with open(filename, 'w') as f:
        f.write('frame_nmr,car_id,car_box,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr, cars in results.items():
            for car_id, data in cars.items():
                car = data.get('car')
                plate = data.get('license_plate')

                if car and plate and 'text' in plate:
                    f.write(f"{frame_nmr},{car_id},"
                            f"[{car['bbox'][0]} {car['bbox'][1]} {car['bbox'][2]} {car['bbox'][3]}],"
                            f"[{plate['bbox'][0]} {plate['bbox'][1]} {plate['bbox'][2]} {plate['bbox'][3]}],"
                            f"{plate['bbox_score']},{plate['text']},{plate['text_score']}\n")
    print(f"[INFO] CSV saved to {filename}")


# =====================================================================
# Kiểm tra định dạng biển số Việt Nam
# =====================================================================
def license_complies_format(text):
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

    patterns = [
        r'^[0-9]{2}[A-Z][0-9]{4,5}$',  # 30A12345
        r'^[A-Z]{2}[0-9]{5}$',         # TM12345
        r'^(NN|NG)[0-9]{3,5}$'         # NN12345
    ]
    return any(re.match(p, text) for p in patterns)


# =====================================================================
# Chuẩn hoá biển số về dạng hợp lệ (ví dụ: 30A12345)
# =====================================================================
def format_license(text):
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    text = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text)

    match_private = re.match(r'^([0-9]{2}[A-Z])([0-9]+)$', text)
    match_other = re.match(r'^([A-Z]{2}|NN|NG)([0-9]+)$', text)

    if match_private:
        return match_private.group(1) + match_private.group(2)
    elif match_other:
        return match_other.group(1) + match_other.group(2)
    else:
        return text


# =====================================================================
# Xác định biển số thuộc về xe nào
# =====================================================================
def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, _, _ = license_plate

    best_car = (-1, -1, -1, -1, -1)
    max_overlap = 0

    for (xc1, yc1, xc2, yc2, car_id) in vehicle_track_ids:
        overlap_x1 = max(x1, xc1)
        overlap_y1 = max(y1, yc1)
        overlap_x2 = min(x2, xc2)
        overlap_y2 = min(y2, yc2)

        overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
        plate_area = (x2 - x1) * (y2 - y1)

        overlap_ratio = overlap_area / plate_area if plate_area > 0 else 0

        if overlap_ratio > 0.6 and overlap_ratio > max_overlap:
            best_car = (xc1, yc1, xc2, yc2, car_id)
            max_overlap = overlap_ratio

    return best_car


# =====================================================================
# Đọc và nhận dạng biển số từ ảnh
# =====================================================================
def read_license_plate(license_plate_crop):
    if license_plate_crop is None or license_plate_crop.size == 0:
        print("[WARN] Empty crop, skip.")
        return None, None

    license_plate_crop = cv2.resize(license_plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # --- Hiển thị ảnh crop ---
    cv2.imshow("License Plate Crop", license_plate_crop)
    cv2.waitKey(1)

    # --- Chuyển sang ảnh xám & xử lý ---
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 19, 9)

    # --- Hiển thị ảnh đã xử lý ---
    cv2.imshow("Threshold Plate", thresh)
    cv2.waitKey(0)  # để xem 1s

    h, w = thresh.shape[:2]
    aspect_ratio = w / h

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 30]

    if len(char_heights) > 0:
        avg_char_h = np.mean(char_heights)
        if avg_char_h < h / 1.8:  # có 2 hàng
            is_double = True
        else:
            is_double = False
    else:
        is_double = aspect_ratio <= 2.5

    if not is_double:
        config = (
            f'--oem 3 --psm 8 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- '
            r'-c tessedit_do_invert=0 '
            r'-c preserve_interword_spaces=1'
        )

        if thresh is None or thresh.size == 0:
            print("[WARN] Empty thresh.")
            return None, None

        text = pytesseract.image_to_string(thresh, config=config)
    else:
        upper = thresh[0:int(h/2), :]
        upper_cleaned = cv2.medianBlur(upper, 3)
        # upper_proc = cv2.bilateralFilter(upper, 9, 75, 75)
        # upper_proc = cv2.equalizeHist(upper_proc)
        # upper_proc = cv2.convertScaleAbs(upper_proc, alpha=1.4, beta=10)
        # upper_proc = cv2.adaptiveThreshold(
        #     upper_proc, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY_INV, 19, 5
        # )

        lower = thresh[int(h/2):, :]

        cv2.imshow("Upper", upper)
        cv2.imshow("Lower", lower)
        cv2.waitKey(0)

        config = (
            f'--oem 3 --psm 7 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- '
            r'-c tessedit_do_invert=0 '
            r'-c preserve_interword_spaces=1'
        )

        text1_raw = pytesseract.image_to_string(upper_cleaned, config=config)
        text2_raw = pytesseract.image_to_string(lower, config=config)

        text1_raw = text1_raw.strip().upper().replace(" ", "").replace("\n", "")
        text2_raw = text2_raw.strip().upper().replace(" ", "").replace("\n", "")

        print(f"[OCR TOP raw]: {text1_raw}")
        print(f"[OCR BOT raw]: {text2_raw}")

        text1_fixed = fix_top_plate(text1_raw)
        text2_fixed = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text2_raw)

        print(f"[FIXED TOP]: {text1_fixed}")
        print(f"[FIXED BOT]: {text2_fixed}")

        merged_text = text1_fixed + text2_fixed
        merged_text_fixed = fix_plate_chars_smart(merged_text)

        print(f"[MERGED RAW]: {merged_text}")
        print(f"[MERGED FIXED]: {merged_text_fixed}")

        text = merged_text_fixed

    # --- OCR ---
    # config = (
    #     f'--oem 3 --psm {psm_mode} '
    #     r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- '
    #     r'-c tessedit_do_invert=0 '
    #     r'-c preserve_interword_spaces=1'
    # )
    # text = pytesseract.image_to_string(thresh, config=config)

    # --- Làm sạch ---
    text = text.strip().upper().replace(" ", "").replace("\n", "")
    text = re.sub(r'[^A-Z0-9]', '', text)
    print(f"[OCR Raw] {text}")

    text = fix_plate_chars_smart(text)
    print(f"[OCR Fixed] {text}")

    # --- Kiểm tra hợp lệ ---
    if license_complies_format(text):
        final_text = format_license_smart(text)
        print(f"[OK] Final Plate: {final_text}")
        return final_text, 1.0
    else:
        print(f"[FAIL] Invalid Plate: {text}")
        return None, None

# def read_license_plate_openalpr(license_plate_crop):
#     if (license_plate_crop.size is None or license_plate_crop.size == 0):
#         print(f"[WARN] Empty crop, skip")
#         return None, None
#
#     temp_path = "temp_plate.jpg"
#     cv2.imwrite(temp_path, license_plate_crop)
#
#     openalpr_exe = os.path.join(os.getcwd(), "openalpr", "alpr.exe")
#
#     command = [
#         openalpr_exe,
#         "-c", "vn",
#         "--topn", "1",
#         temp_path
#
#     ]
#
#     try:
#         result = subprocess.run(command, capture_output=True, text=True, timeout=5)
#         output = result.stdout
#     except Exception as e:
#         print(f"[ERROR] OpenALPR failed: {e}")
#         return None, None
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#
#     plate, confidence = None, 0.0
#     for line in output.splitlines():
#         if "plate" in line and "confidence" in line:
#             parts = line.split()
#             if len(parts) >= 3:
#                 plate = parts[1]
#                 try:
#                     confidence = float(parts[-1])
#                 except:
#                     confidence = 0.0
#                 break
#
#     if plate:
#         print(f"[ALPR] {plate} (conf={confidence:.2f})")
#         return plate, confidence / 100  # đưa confidence về [0,1]
#     else:
#         print("[ALPR] No plate detected.")
#         return None, None
