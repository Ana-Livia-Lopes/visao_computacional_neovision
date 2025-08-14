import torch
import cv2
import time
from collections import defaultdict
import pyttsx3


voz = pyttsx3.init()
voz.setProperty('rate', 150)

# voz pt-BR
for v in voz.getProperty('voices'):
    if 'brazil' in v.name.lower() or 'português' in v.name.lower():
        voz.setProperty('voice', v.id)
        break

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
model.classes = None
class_names = model.names

traducao = {
    'person': 'pessoa',
    'bottle': 'garrafa',
    'chair': 'cadeira',
    'backpack': 'mochila',
    'cell phone': 'celular',
    'laptop': 'notebook',
}

# Abrir a câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

INTERVALO = 5
CONFIANCA_MINIMA = 0.6

try:
    while cap.isOpened():
        time.sleep(INTERVALO)
        ret, frame = cap.read() # aqui ta lendo apenas um frame da camera

        # detectar
        results = model(frame)
        detections = results.pandas().xyxy[0]

        contagem = defaultdict(int)

        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            if conf >= CONFIANCA_MINIMA:
                contagem[label] += 1

        if contagem:
            frases = []
            for nome, qtd in contagem.items():
                nome_pt = traducao.get(nome, nome)
                if qtd == 1:
                    frases.append(f"{qtd} {nome_pt}")
                else:
                    if nome_pt.endswith('r') or nome_pt.endswith('a'):
                        nome_plural = nome_pt + 's'
                    else:
                        nome_plural = nome_pt
                    frases.append(f"{qtd} {nome_plural}")

            frase_final = "Detectado: " + ", ".join(frases)
            print("---> " + frase_final)
            voz.say(frase_final)
            voz.runAndWait()
        else:
            frase_final = "Nenhum objeto encontrado."
            print("---> " + frase_final)
            voz.say(frase_final)
            voz.runAndWait()

except KeyboardInterrupt:
    print("\nEncerrando...")

cap.release()
