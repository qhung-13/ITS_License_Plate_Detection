import numpy as np
import re
import cv2
import pytesseract

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
    'Q': '9'
}

LETTER_FIX_MAP = {
    '0': 'O', '6': 'G', '1': 'I', '5': 'S', '2': 'Z'
}

def fix_plate_chars_smart(text):
    text = text.upper()
    # Biển số VN dạng 30A12345
    if len(text) >= 7:
        head = text[:2]  # 2 số đầu
        letter = text[2]  # chữ cái
        tail = text[3:]  # phần số còn lại

        # Map chữ cái nếu OCR nhầm thành số
        letter = LETTER_FIX_MAP.get(letter, letter)

        # Map các ký tự nhầm trong phần số
        tail = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in tail)

        return head + letter + tail
    return text

def format_license_smart(text):
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

    if len(text) >= 7:
        head = text[:2]
        letter = text[2]
        tail = text[3:]
        return head + letter + tail  # giữ nguyên đã fix trong fix_plate_chars_smart
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

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_heights = [cv2.boundingRect(c)[3] for c in contours if cv2.contourArea(c) > 30]
    h, w = thresh.shape[:2]
    aspect_ratio = w / h
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
        text = pytesseract.image_to_string(thresh, config=config)
    else:
        upper = thresh[0:int(h/2), :]

        upper_proc = cv2.bilateralFilter(upper, 9, 75, 75)
        upper_proc = cv2.equalizeHist(upper_proc)
        upper_proc = cv2.convertScaleAbs(upper_proc, alpha=1.4, beta=10)
        upper_proc = cv2.adaptiveThreshold(
            upper_proc, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 5
        )

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

        text1 = pytesseract.image_to_string(upper_proc, config=config)
        text2 = pytesseract.image_to_string(lower, config=config)
        text = (text1 + text2)

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

    # --- Kiểm tra hợp lệ ---
    if license_complies_format(text):
        final_text = format_license_smart(text)
        print(f"[OK] Final Plate: {final_text}")
        return final_text, 1.0
    else:
        print(f"[FAIL] Invalid Plate: {text}")
        return None, None
