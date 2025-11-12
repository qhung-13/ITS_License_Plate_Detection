from ultralytics import YOLO
import multiprocessing

def main():
    multiprocessing.freeze_support()

    # Load mô hình đã train trước đó
    model = YOLO(r"runs/detect/License_Plate_Finetune_v10/weights/best.pt")

    model.train(
        data=r"DataSet/LP_detection.v1i.yolov11/data.yaml",
        epochs=60,
        imgsz=768,
        batch=4,
        lr0=0.0001,
        optimizer="AdamW",
        pretrained=True,
        device=0,
        name="License_Plate_Model-v11",

        mosaic=0.0,
        mixup=0.0,
        translate=0.01,
        scale=0.03,
        degrees=0.5,
        shear=0.0,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,

        patience=20,
        deterministic=True,
        workers=1,
        amp=True,
        dropout=0.05,
        close_mosaic=0,

        freeze=0,
    )

if __name__ == "__main__":
    main()
