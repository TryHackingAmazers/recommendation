from ultralytics import YOLO

def train():
    dataset_config_path = "./datasets/furniture-detection-20/data.yaml"
    model = YOLO("yolov8n.pt")
    model.train(data=dataset_config_path, epochs=10)

def load():
    model = YOLO("/home/rohan/hackonama/recommendation/checkpoint/best.pt")
    return model