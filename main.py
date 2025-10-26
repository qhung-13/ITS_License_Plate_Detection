from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from until import get_car, read_license_plate, write_csv

# --------- INIT ---------------
results = {}
mot_tracker = Sort()

#---------- Load Models --------------
coco_model = YOLO('Yolo-Weights/yolov8n.pt')
license_plate_detector = YOLO(r"runs\detect\train_v8_plate3\weights\best.pt")

#------------ Load Video ----------------
cap = cv2.VideoCapture(r"Data\datatest1.mp4")

vehicles = [2, 3, 5, 7]

classNames = ["plate"]

# ------------- Read Frames --------------
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    #--------Detect Vehicles -------------------
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    #-------------- Track Vehicles ---------------------
    track_ids = mot_tracker.update(np.asarray(detections_))

    for x1, y1, x2, y2, car_id in track_ids:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Car {int(car_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #---------------- Detect License Plate --------------
    license_plates = license_plate_detector(frame)[0]

    print(f"Frame {frame_nmr}: detected {len(license_plates.boxes)} plates")

    for box in license_plates.boxes:
        print(box.data)

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # ------------ Assign License Plate For Car -------------
        xcar1, ycar1, xcar2,ycar2, car_id = get_car(license_plate, track_ids)
        print(f"→ matched car_id = {car_id}")
        if car_id == -1:
            continue

        #------------- Crop License Plate--------------------
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        # h, w, _ = frame.shape
        # x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        # license_plate_crop = frame[y1:y2, x1:x2]
        cv2.imshow("license_plate_crop", license_plate_crop)
        cv2.waitKey(1)

        if license_plate_crop.size == 0:
            continue

        # ----------------- Process License Plate -------------------
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        license_plate_crop_gray = cv2.equalizeHist(license_plate_crop_gray)
        license_plate_crop_gray = cv2.convertScaleAbs(license_plate_crop_gray, alpha=1.3, beta=15)
        license_plate_crop_gray = cv2.medianBlur(license_plate_crop_gray, 3)

        license_plate_crop_thresh = cv2.adaptiveThreshold(
            license_plate_crop_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            19, 9
        )

        #------------------ Read License Plate Number ------------------
        # license_plate_crop_thresh = cv2.bitwise_not(license_plate_crop_thresh)
        # cv2.imshow("OCR_inputt", license_plate_crop_thresh)

        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

        # ------------------ Save Results ------------------
        if license_plate_text is not None:
            results[frame_nmr][car_id] = {'car' : {'bbox' : [xcar1, ycar1, xcar2, ycar2],},
                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                            'text': license_plate_text,
                                                            'bbox_score': score,
                                                            'text_score': license_plate_text_score}}

            # --- Draw on Frame ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, license_plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ================== DISPLAY ===================
    cv2.namedWindow("vehicles _ License Plate Tracking", cv2.WINDOW_NORMAL)

    screen_w, screen_h = 1280, 720
    h, w, _ = frame.shape
    scale = min(screen_w / w, screen_h / h)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("vehicles _ License Plate Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ----------------- Write Results --------------------------
write_csv(results, './test.csv')




