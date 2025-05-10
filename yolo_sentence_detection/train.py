from ultralytics import YOLO

# Load YOLOv8x pretrained on COCO
model = YOLO("yolov8x.pt")

# Train on your character detection dataset
model.train(
    data="dataset/dataset.yaml",  # 🔁 path to your dataset.yaml
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    optimizer="AdamW",
    dropout=0.1,
    amp=True,
    val=True,
    plots=True,

    # Data Augmentation
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,

    project="runs/detect",
    name="char_detector_yolov8x",
    pretrained=True
)
