from ultralytics import YOLO
import cv2
from until import read_license_plate

# 1. Load model YOLO đã train để phát hiện biển số
model = YOLO(r"runs/detect/License_Plate_Models-v5/weights/best.pt")  # thay bằng đường dẫn model của bạn

# 2. Đọc ảnh full xe
img = cv2.imread("CarLongPlate74_jpg.rf.db60f9eb830a7a74b1bf957d817d62b9.jpg")

# 3. Dự đoán bounding box biển số
results = model.predict(img)

# 4. Lặp qua từng kết quả
for r in results:
    for box in r.boxes.xyxy:  # mỗi box là x1, y1, x2, y2
        x1, y1, x2, y2 = map(int, box)
        plate_img = img[y1:y2, x1:x2]  # cắt vùng biển số
        number = read_license_plate(plate_img)  # đọc ký tự
        print("Detected plate:", number)

        # Vẽ khung màu đỏ quanh biển số
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Ghi biển số lên ảnh
        cv2.putText(img, number, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 5. Hiển thị ảnh kết quả
cv2.imshow("License Plate Detection", img)
cv2.waitKey(1)
cv2.destroyAllWindows()
