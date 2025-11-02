from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from until import get_car, read_license_plate, write_csv

# =========================================================
# INIT
# =========================================================
results = {}
mot_tracker = Sort()

# =========================================================
# Load Models
# =========================================================
coco_model = YOLO('Yolo-Weights/yolo11m.pt')
license_plate_detector = YOLO(r"runs/detect/License_Plate_Models-v8/weights/best.pt")

# =========================================================
# Load Video
# =========================================================
cap = cv2.VideoCapture(r"Data/datatest1.mp4")
if not cap.isOpened():
    raise RuntimeError("❌ Không mở được video! Kiểm tra lại đường dẫn.")

vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck
frame_nmr = -1

# =========================================================
# MAIN LOOP
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}

    # -------------------------------------------------------
    # Detect Vehicles
    # -------------------------------------------------------
    detections = coco_model(frame, verbose=False)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # -------------------------------------------------------
    # Track Vehicles
    # -------------------------------------------------------
    track_ids = mot_tracker.update(np.asarray(detections_))
    for x1, y1, x2, y2, car_id in track_ids:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Car {int(car_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -------------------------------------------------------
    # Detect License Plates
    # -------------------------------------------------------
    license_plates = license_plate_detector(frame, verbose=False)[0]
    print(f"[Frame {frame_nmr}] Detected {len(license_plates.boxes)} license plates")

    # -------------------------------------------------------
    # Match Each License Plate with Vehicle
    # -------------------------------------------------------
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id == -1:
            continue

        # ---------------- Crop & Preprocess Plate ----------------
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        license_plate_crop = frame[y1:y2, x1:x2]

        if license_plate_crop.size == 0:
            continue
        # --- Hiển thị ảnh crop ---
        cv2.imshow("Cropped Plate", license_plate_crop)
        cv2.waitKey(1)

        # ---------------- OCR ----------------
        plate_text, plate_text_score = read_license_plate(license_plate_crop)

        if plate_text is not None:
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': plate_text,
                    'bbox_score': score,
                    'text_score': plate_text_score
                }
            }

            # Draw result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # -------------------------------------------------------
    # 5️⃣ Display
    # -------------------------------------------------------
    cv2.namedWindow("Vehicle & License Plate Tracking", cv2.WINDOW_NORMAL)
    screen_w, screen_h = 1280, 720
    h, w, _ = frame.shape
    scale = min(screen_w / w, screen_h / h)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("Vehicle & License Plate Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# END – Save Results
# =========================================================
cap.release()
cv2.destroyAllWindows()
write_csv(results, './test.csv')
print("✅ Done. Saved results to test.csv")
