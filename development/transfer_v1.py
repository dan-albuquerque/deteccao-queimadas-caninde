import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

api_key = os.getenv("ROBOFLOW_KEY")
workspace = os.getenv("WORKSPACE")
project_name = os.getenv("PROJECT")
version_num = int(os.getenv("VERSION"))
download_version = os.getenv("DOWNLOAD_VERSION")

rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project_name)
version = project.version(version_num)
dataset = version.download(download_version)

model = YOLO("yolov8n.pt")

# treinamento

model.train(
    data='projetos6/Forest-fire-1/data.yaml',
    imgsz=416,  # Tamanho da imagem reduzido
    epochs=70,
    batch=8,
    workers=6, 
    device=0    # usa GPU
)
