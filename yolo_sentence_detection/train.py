from ultralytics import YOLO

# Load a pretrained YOLOv8s model
model = YOLO("yolov8s.pt")

# Train the model using your dataset YAML file
model.train(
    data="dataset/dataset.yaml",  # Path to dataset.yaml
    epochs=50,
    imgsz=640,
    batch=16,        # You can adjust based on your GPU memory
    name="yolo_sentence_detector"
)
