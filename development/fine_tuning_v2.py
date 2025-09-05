from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="0wXzyJxDPq48zzcy9ZWX")
project = rf.workspace("testes-yolo").project("fires-ttnhi-b63rs")
version = project.version(1)
dataset = version.download("yolov8")

salvar_modelo = '/home/danilo/projetosMeus/projetos6/runs/fine_tuneV4'
dataset_novo = '/home/danilo/projetosMeus/projetos6/fires-1/data.yaml'
modelo2tune = '/home/danilo/projetosMeus/projetos6/runs/fine_tune/train/weights/best_v3.pt'

modelo = YOLO(modelo2tune)

modelo.train(
    data=dataset_novo,
    imgsz=416,
    epochs=30,       
    batch=8,
    workers=6,
    project=salvar_modelo,
    device=0
)
