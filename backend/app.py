# backend_production_safe.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import threading
from datetime import datetime

app = FastAPI(title="Vehicle Tracking API")

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Folder chứa các file JSON từ nhiều cam -----------------
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "multi_cam_data"))

# ----------------- Memory cache -----------------
RECORDS = {}  # {plate: [list of records]}

# ----------------- Utils -----------------
def normalize_plate(plate: str) -> str:
    """Chuẩn hóa plate trước khi lookup"""
    return plate.upper().replace(" ", "").replace("-", "").replace(".", "")

def load_all_records():
    global RECORDS
    RECORDS.clear()
    if not os.path.exists(DATA_FOLDER):
        print(f"[WARN] DATA_FOLDER not exist: {DATA_FOLDER}")
        return

    for root, _, files in os.walk(DATA_FOLDER):  # walk cả subfolder
        for fname in files:
            if fname.endswith(".json"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for plate, entries in data.items():
                            plate_norm = normalize_plate(plate)
                            RECORDS.setdefault(plate_norm, []).extend(entries)
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")

    print(f"[INFO] Loaded {len(RECORDS)} plates from JSON files.")

# ----------------- Thread-safe reload -----------------
reload_lock = threading.Lock()
def reload_records():
    with reload_lock:
        load_all_records()

# ----------------- Initial load -----------------
load_all_records()

# ----------------- API -----------------
@app.get("/")
def root():
    return {"message": "Vehicle Tracking API is running"}

@app.get("/vehicle/{plate}")
def get_vehicle_route(plate: str):
    plate_norm = normalize_plate(plate)

    with reload_lock:
        records = RECORDS.get(plate_norm)

    if not records:
        raise HTTPException(status_code=404, detail=f"No records found for plate {plate}")

    # Sort theo timestamp
    try:
        records_sorted = sorted(
            records,
            key=lambda r: datetime.fromisoformat(r["timestamp"].replace("Z", ""))
        )
    except Exception:
        records_sorted = records  # fallback nếu lỗi format timestamp

    return {"plate": plate_norm, "route": records_sorted}

# ----------------- Optional: reload API endpoint -----------------
@app.post("/reload")
def reload_api():
    """Endpoint để reload tất cả file JSON từ folder"""
    reload_records()
    return {"message": "Records reloaded", "total_plates": len(RECORDS)}
