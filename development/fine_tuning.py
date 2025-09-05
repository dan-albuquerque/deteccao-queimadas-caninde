from ultralytics import YOLO

salvar_modelo = '/home/danilo/projetosMeus/projetos6/runs/fine_tuneV2'
dataset_antigo = '/home/danilo/projetosMeus/projetos6/Forest-fire-2/data.yaml'
pesos_v2 = '/home/danilo/projetosMeus/projetos6/runs/detect/train2/weights/best.pt'

modelo = YOLO(pesos_v2)

modelo.train(
    data=dataset_antigo,
    imgsz=416,
    epochs=30,       
    batch=8,
    workers=6,
    project=salvar_modelo,
    device=0
)
