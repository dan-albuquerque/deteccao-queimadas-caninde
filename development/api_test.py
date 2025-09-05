from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Carregar o modelo treinado
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler bytes da imagem enviada
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Rodar inferência
    results = model.predict(img, imgsz=416, conf=0.5)

    # Extrair bounding boxes [x1, y1, x2, y2, conf, class]
    detections = results[0].boxes.data.tolist()

    return {"detections": detections}
