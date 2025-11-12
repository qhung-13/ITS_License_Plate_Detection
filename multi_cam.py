from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from until import get_car, read_license_plate_bike, read_license_plate_car, append_json, write_csv
from config import vehical_types
import os

CONF_THRESHOLD = 0.5
VEHICLES = [2, 3, 5, 7]  # car, motorbike, bus, truck
OUTPUT_FOLDER = "./multi_cam_data"

# =========================================================
# Fake lộ trình Tô Kí -> Võ Oanh
# =========================================================
FAKE_ROUTE = [
    {"cam_id": "CAM-1", "location": "Tô Kí", "lat": 10.8470, "lon": 106.6730},
    {"cam_id": "CAM-2", "location": "Quang Trung", "lat": 10.8460, "lon": 106.6745},
    {"cam_id": "CAM-3", "location": "Nguyễn Văn Nghi", "lat": 10.8450, "lon": 106.6755},
    {"cam_id": "CAM-4", "location": "Phạm Văn Đồng", "lat": 10.8440, "lon": 106.6760},
    {"cam_id": "CAM-5", "location": "Nguyễn Xí", "lat": 10.8430, "lon": 106.6765},
    {"cam_id": "CAM-6", "location": "Ung Văn Khiêm", "lat": 10.8420, "lon": 106.6770},
    {"cam_id": "CAM-7", "location": "Nguyễn Gia Trí", "lat": 10.8410, "lon": 106.6775},
    {"cam_id": "CAM-8", "location": "Võ Oanh", "lat": 10.8400, "lon": 106.6780},
]


# =========================================================
# Hàm xử lý video cho từng cam
# =========================================================
def process_video(cam_id, video_path, cam_info=None):
    print(f"[{cam_id}] Bắt đầu xử lý video: {video_path}")

    # Tracker & YOLO model (mỗi cam tạo riêng)
    mot_tracker = Sort()
    coco_model = YOLO('Yolo-Weights/yolo11m.pt')  # có thể thay bằng yolov8n.pt cho nhẹ
    license_plate_detector = YOLO(r"runs/detect/License_Plate_Finetune_v10/weights/best.pt")

    # Folder lưu kết quả
    cam_folder = os.path.join(OUTPUT_FOLDER, cam_id)
    os.makedirs(cam_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{cam_id}] Không mở được video: {video_path}")
        return {}

    frame_nmr = -1
    results_dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1
        results_dict[frame_nmr] = {}

        # --- Detect Vehicles ---
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in VEHICLES and score >= CONF_THRESHOLD:
                detections_.append([x1, y1, x2, y2, score, int(class_id)])

        # --- Track Vehicles ---
        track_ids = mot_tracker.update(np.array([d[:5] for d in detections_]))
        track_type_map = {}
        for idx, (x1, y1, x2, y2, track_id) in enumerate(track_ids):
            class_id = int(detections_[idx][5])
            class_type = vehical_types.get(class_id, 'car')
            track_type_map[int(track_id)] = class_type

        # --- Detect License Plates ---
        license_plates = license_plate_detector(frame, verbose=False)[0]

        for lp in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = map(int, lp[:6])
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
            if car_id == -1:
                continue

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            license_plate_crop = frame[y1:y2, x1:x2]
            if license_plate_crop.size == 0:
                continue

            v_type = track_type_map.get(int(car_id), "car")
            if v_type == "bike":
                plate_text, plate_text_score = read_license_plate_bike(license_plate_crop)
            else:
                plate_text, plate_text_score = read_license_plate_car(license_plate_crop)

            if plate_text:
                results_dict[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': plate_text,
                        'bbox_score': score,
                        'text_score': plate_text_score
                    }
                }

                # --- Gán location + lat/lon ---
                if cam_info:
                    location = cam_info.get("location", cam_id)
                    lat = cam_info.get("lat", None)
                    lon = cam_info.get("lon", None)
                else:
                    location = cam_id
                    lat = None
                    lon = None

                append_json(
                    plate_text=plate_text,
                    v_type=v_type,
                    car_id=car_id,
                    cam_id=cam_id,
                    location=location,
                    lat=lat,
                    lon=lon,
                    base_folder=cam_folder
                )

    cap.release()
    print(f"[{cam_id}] Xử lý xong video: {video_path}")
    return results_dict


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_results = {}

    # Dùng FAKE_ROUTE tạm
    cams_info = {r["cam_id"]: r for r in FAKE_ROUTE}

    for cam_id, cam_info in cams_info.items():
        video_path = "Data/datatest1.mp4"  # đổi sang video thật nếu có
        cam_results = process_video(cam_id, video_path, cam_info)
        all_results[cam_id] = cam_results

    # Lưu kết quả tổng hợp
    write_csv(all_results, os.path.join(OUTPUT_FOLDER, 'multi_cam_results.csv'))
    write_csv(all_results, os.path.join(OUTPUT_FOLDER, 'multi_cam_results.json'))
    print(f"Xong tất cả cam. Kết quả lưu trong folder {OUTPUT_FOLDER}")

