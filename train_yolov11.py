from ultralytics import YOLO
import multiprocessing

def main():
    multiprocessing.freeze_support()

    model = YOLO(r"Yolo-Weights/yolo11n.pt")

    model.train(
        data=r"DataSet/object detection.v1i.yolov11/data.yaml",
        epochs=80,
        imgsz=640,
        batch=4,
        lr0=0.001,
        optimizer="AdamW",
        mosaic=0.8,
        mixup=0.2,
        translate=0.1,
        scale=0.3,
        degrees=10,
        shear=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        patience=40,
        device=0,
        name="License_Plate_Models",
        pretrained=True,
        deterministic=True,
        workers=2,
    )

if __name__ == "__main__":
    main()
