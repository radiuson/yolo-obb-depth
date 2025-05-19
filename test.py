from ultralytics import YOLO
import torch


model = YOLO("/home/ihpc/code/yolo/yolo-obb-depth/ultralytics_rep/ultralytics/cfg/models/v8/yolov8-obb-depth.yaml",task="obb-depth")  # Load a model
x = torch.randn(16, 3, 640, 640)
print(model)
y = model(x)  # Inference
print(y)  # Predictions