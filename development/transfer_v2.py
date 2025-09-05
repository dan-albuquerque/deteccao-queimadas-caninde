from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="0wXzyJxDPq48zzcy9ZWX")
project = rf.workspace("testes-yolo").project("forest-fire-xawoh-g43ew")
version = project.version(2)
dataset = version.download("yolov8")

model = YOLO("yolov8n.pt")


# treinamento
model.train(
    data='/home/danilo/projetosMeus/projetos6/Forest-fire-2/data.yaml',
    imgsz=416, # Tamanho da imagem reduzido
    epochs=70,
    batch=8,
    workers=6,
    device=0
)
