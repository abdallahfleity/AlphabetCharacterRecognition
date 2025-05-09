from ultralytics import YOLO

# Load the pretrained YOLOv8 model
model = YOLO("runs/detect/char_segment_finetuned/weights/best.pt")  # Or use a fine-tuned model path like "runs/detect/char_segment_finetuned/weights/best.pt"

# Start training with augmentation and monitoring
model.train(
    data="dataset/dataset.yaml",  # Path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    optimizer="AdamW",
    dropout=0.1,
    amp=True,
    val=True,
    plots=True,
    cache=True,

    # 🔁 Data Augmentation
    mosaic=1.0,  # Enables mosaic augmentation
    mixup=0.2,  # Mixes multiple images together (use small value)
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,  # Saturation augmentation
    hsv_v=0.4,  # Value augmentation
    flipud=0.0,  # Vertical flip probability (keep 0 for letters)
    fliplr=0.5,  # Horizontal flip probability

    project="runs/detect",
    name="char_segment_finetuned",
    pretrained=True
)
