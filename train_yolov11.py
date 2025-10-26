from ultralytics import YOLO
import multiprocessing

def main():
    multiprocessing.freeze_support()

    model = YOLO(r"runs/detect/train_v8_plate_pro2/weights/best.pt")

    model.train(
        data=r"LP detection.v1i.yolov8\data.yaml",
        epochs=150,
        imgsz=960,
        batch=16,
        lr0=8e-4,
        optimizer="AdamW",
        mosaic=1.0,
        mixup=0.3,
        translate=0.2,
        scale=0.5,
        degrees=10,
        shear=0.15,
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.6,
        patience=80,
        device=0,
        name="train_v8_plate_5",
        pretrained=True,
        deterministic=True,
    )

if __name__ == "__main__":
    main()
