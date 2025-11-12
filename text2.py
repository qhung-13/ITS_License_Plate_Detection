# import cv2
# import pytesseract
# import numpy as np
# import re
#
# def fix_top_plate(text):
#     text = re.sub(r'[^A-Z0-9]', '', text.upper())
#
#     if(len(text) < 2):
#         return text
#
#     CHAR_FIX_MAP = {
#         'O': '0', 'D': '0', 'Q': '0',
#         'I': '1', 'L': '1',
#         'S': '5',
#         'B': '8',
#         'G': '6',
#         'Z': '2',
#         'E': '8',
#         'T': '7',
#         'R': '4',
#     }
#
#     LETTER_FIX_MAP = {
#         '0': 'O',
#         '6': 'G',
#         '1': 'I',
#         '5': 'S',
#         '2': 'Z',
#         '8': 'B',
#         '4': 'A'
#     }
#
#     PROVINCE_FIX_MAP = {
#         'O': '8',  # O thường bị OCR đọc nhầm thay vì 8
#         'D': '0',
#         'Q': '0',
#         'E': '4',
#         'Z': '2',
#         'S': '5',
#         'T': '7',
#         'B': '8',
#     }
#
#     CHAR_NOT_IN_PLATE = ['I', 'J', 'O', 'Q', 'R', 'W']
#
#     FIX_THIRD_CHAR = {
#         'R': 'A'
#     }
#
#     #----- 2 ki tu dau
#     fixed_first_two = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in text[:2])
#     print(f"[OCR Fixed First] {fixed_first_two}")
#     if (fixed_first_two[0] == '0'):
#         fixed_first_two = ''.join(PROVINCE_FIX_MAP.get(ch, ch) for ch in text[:2])
#         print(f"[OCR Fixed Second] {fixed_first_two}")
#     if (not fixed_first_two.isdigit()):
#         fixed_first_two = ''.join(PROVINCE_FIX_MAP.get(ch, ch) for ch in text[:2])
#         print(f"[OCR Fixed Third] {fixed_first_two}")
#
#     #---- ki tu thu 3
#     third = ''
#     rest = ''
#
#     if len(text) > 2:
#         third = text[2]
#         if third.isdigit():
#             third = LETTER_FIX_MAP.get(third, 'A')
#         else:
#             third = LETTER_FIX_MAP.get(third, third)
#             if (third in CHAR_NOT_IN_PLATE):
#                 third = FIX_THIRD_CHAR.get(third, third)
#
#         rest = text[3:]
#
#     rest_fixed = ''.join(CHAR_FIX_MAP.get(ch, ch) for ch in rest)
#
#     return (fixed_first_two + third + rest_fixed).strip()
#
# pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
#
# config = (
#     f'--oem 3 --psm 8 '
#     r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- '
#     r'-c tessedit_do_invert=0 '
#     r'-c preserve_interword_spaces=1'
# )
#
# img = cv2.imread(r"Screenshot 2025-11-06 230052.png")
#
# # kernel = np.ones((4, 4), np.uint8)
# # thresh_cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
#
# thresh_cleaned = cv2.medianBlur(img, 3)
#
# cv2.imshow("Original Image (Correct for Tesseract)", thresh_cleaned)
# cv2.waitKey(0)
#
# text1_raw = pytesseract.image_to_string(thresh_cleaned, config=config)
# text1_final = fix_top_plate(text1_raw)
# print(f"Kết quả đọc: '{text1_final.strip()}'")
#
# cv2.destroyAllWindows()

import os

# ==========================
# ⚙️ Cấu hình
# ==========================
base_dir = r"DataSet/LP_detection.v1i.yolov11"         # thư mục chứa train/, valid/, test/
splits = ["train", "valid"]

keep_class = 0            # 0 cho BSD, 1 cho BSV
# ==========================

for split in splits:
    lbl_dir = os.path.join(base_dir, split, "labels")
    img_dir = os.path.join(base_dir, split, "images")

    count_total = 0
    count_keep = 0

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        count_total += 1
        new_lines = [l for l in lines if l.startswith(f"{keep_class} ")]

        if new_lines:
            with open(lbl_path, "w") as f:
                # đổi class id về 0 vì giờ chỉ còn 1 class
                for line in new_lines:
                    f.write("0" + line[1:])
            count_keep += 1
        else:
            # nếu không còn dòng nào thì xóa file label và ảnh tương ứng
            os.remove(lbl_path)
            img_name = lbl_file.replace(".txt", ".jpg")
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, lbl_file.replace(".txt", ".png"))
            if os.path.exists(img_path):
                os.remove(img_path)

    print(f"✅ {split}: giữ lại {count_keep}/{count_total} ảnh có class {keep_class}")

print("\n🎯 Đã lọc xong — dataset giờ chỉ còn 1 class duy nhất!")
