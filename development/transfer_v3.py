import os
from ultralytics import YOLO


salvar_modelo = '/home/danilo/projetosMeus/projetos6/runs/fine_tuneV3'
dataset_antigo_v2 = '/home/danilo/projetosMeus/projetos6/Forest-fire-2/data.yaml'
pesos_v1 = '/home/danilo/projetosMeus/projetos6/runs/detect/train2/weights/best.pt'


model = YOLO(pesos_v1)

model.train(
    data=dataset_antigo_v2,
    imgsz=416,
    epochs=40,
    batch=8,                # dobrar batch size melhora estabilidade e uso da GPU
    workers=8,               # mais workers acelera o carregamento de dados
    project=salvar_modelo ,  # salvar resultados aqui
    device=0,
    lr0=0.01,                # learning rate inicial um pouco maior acelera aprendizagem
    lrf=0.1,                 # decaição suave do lr
    optimizer="SGD",         # SGD geralmente converge melhor em detecção
    momentum=0.937,          # valor testado para estabilidade
    weight_decay=0.0005,     # evitar overfitting
    warmup_epochs=3,         # warmup para estabilizar no início
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # augmentações de cor ajudam em variabilidade de iluminação
    degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
    mosaic=1.0, mixup=0.5,  # amplia diversidade de dados e evita sobreajuste
    resume=False             # começar do zero para controle total
)