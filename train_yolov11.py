from ultralytics import YOLO
import multiprocessing

def main():
    multiprocessing.freeze_support()

    # model = YOLO("Yolo-Weights/yolo11m.pt")
    model = YOLO(r"runs/detect/License_Plate_Models-v7/weights/best.pt")

    model.train(
        data=r"DataSet/Licence Plate 3.v1i.yolov11/data.yaml",
        epochs=30,
        imgsz=768,
        batch=1,
        lr0=0.0005,
        optimizer="AdamW",
        mosaic=0.2,
        mixup=0.1,
        translate=0.05,
        scale=0.2,
        degrees=5,
        shear=0.05,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        patience=20,
        device=0,
        name="License_Plate_Models-v8",
        pretrained=True,
        deterministic=True,
        workers=1,
        amp=True,
        dropout=0.1,
        close_mosaic=3,
    )


if __name__ == "__main__":
    main()
