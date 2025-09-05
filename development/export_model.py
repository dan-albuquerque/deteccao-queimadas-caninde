from ultralytics import YOLO

model = YOLO("projetos6/runs/detect/train2/weights/best.pt")
model.export(format="onnx", opset=12, simplify=False)
