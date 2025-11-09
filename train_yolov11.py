from ultralytics import YOLO
import multiprocessing

def main():
    multiprocessing.freeze_support()

    model = YOLO(r"runs/detect/License_Plate_Models-v9/weights/best.pt")

    model.train(
        data=r"DataSet/LP detection.v1i.yolov11/data.yaml",
        epochs=20,
        imgsz=768,
        batch=4,
        lr0=0.00025,
        optimizer="AdamW",
        pretrained=True,
        device=0,
        name="License_Plate_Finetune_v10",

        mosaic=0.05,
        mixup=0.0,
        translate=0.02,
        scale=0.05,
        degrees=1,
        shear=0.01,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,

        patience=15,
        deterministic=True,
        workers=1,
        amp=True,
        dropout=0.05,
        close_mosaic=3,

        freeze=10,  # freeze 10 layer đầu
    )

if __name__ == "__main__":
    main()
