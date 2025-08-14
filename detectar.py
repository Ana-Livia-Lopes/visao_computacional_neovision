import torch
import cv2
import time
from collections import defaultdict
from gtts import gTTS
import pygame
import io
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pygame.mixer.init()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
model.classes = None

# Tradução dos objetos
traducao = {
    'person': 'pessoa',
    'bottle': 'garrafa',
    'chair': 'cadeira',
    'backpack': 'mochila',
    'cell phone': 'celular',
    'laptop': 'notebook',
}

# Gênero dos objetos (f = feminino, m = masculino)
genero = {
    'person': 'f',
    'bottle': 'f',
    'chair': 'f',
    'backpack': 'f',
    'cell phone': 'm',
    'laptop': 'm',
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

INTERVALO = 10
CONFIANCA_MINIMA = 0.55
ultimo_tempo_fala = 0
ultima_frase = ""

def falar(texto):
    mp3_fp = io.BytesIO()
    tts = gTTS(texto, lang='pt')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        contagem = defaultdict(int)

        for _, row in detections.iterrows():
            if row['confidence'] >= CONFIANCA_MINIMA:
                contagem[row['name']] += 1

        agora = time.time()

        if agora - ultimo_tempo_fala >= INTERVALO:
            if contagem:
                frases = []
                for nome, qtd in contagem.items():
                    nome_pt = traducao.get(nome, nome)
                    gen = genero.get(nome, 'm')  

                    if qtd == 1:
                        artigo = 'uma' if gen == 'f' else 'um'
                        verbo = 'detectada' if gen == 'f' else 'detectado'
                    else:
                        if qtd == 2:
                            artigo = 'duas' if gen == 'f' else 'dois'
                        else:
                            artigo = str(qtd)
                        verbo = 'detectadas' if gen == 'f' else 'detectados'
                        if nome_pt.endswith(('r', 'a', 'k')):
                            nome_pt += 's'

                    frases.append(f"{artigo} {nome_pt} {verbo}")

                frase_final = ", ".join(frases)
            else:
                frase_final = "Nenhum objeto encontrado"

            if frase_final != ultima_frase:
                print("---> " + frase_final)
                falar(frase_final)
                ultima_frase = frase_final
                ultimo_tempo_fala = agora

except KeyboardInterrupt:
    print("\nEncerrando...")

cap.release()
pygame.mixer.quit()
