
from ultralytics import YOLO
import os

DATA_PATH = r"C:\Users\USER\Desktop\bottle_dataset\data.yaml"


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"problem var")


model = YOLO("yolov8n.pt")

model.train(
    data=DATA_PATH,
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu'  
)

model.save("bottle_defect_model.pt")
print("Model save")

