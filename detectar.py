import torch
import cv2
from collections import defaultdict
from gtts import gTTS
import pygame
import io
import warnings
import sys
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
pygame.mixer.init()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
model.classes = None

# Tradução dos objetos (ta faltando ainda tenho que arrumar)
traducao = {
    'person': 'pessoa',
    'bicycle': 'bicicleta',
    'car': 'carro',
    'motorcycle': 'moto',
    'airplane': 'avião',
    'bus': 'ônibus',
    'train': 'trem',
    'truck': 'caminhão',
    'boat': 'barco',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'placa de pare',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pássaro',
    'cat': 'gato',
    'dog': 'cachorro',
    'horse': 'cavalo',
    'sheep': 'ovelha',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'urso',
    'zebra': 'zebra',
    'giraffe': 'girafa',
    'backpack': 'mochila',
    'umbrella': 'guarda-chuva',
    'handbag': 'bolsa',
    'tie': 'gravata',
    'suitcase': 'mala',
    'frisbee': 'frisbee',
    'skis': 'esquis',
    'snowboard': 'snowboard',
    'sports ball': 'bola',
    'kite': 'pipa',
    'baseball bat': 'taco de beisebol',
    'baseball glove': 'luva de beisebol',
    'skateboard': 'skate',
    'surfboard': 'prancha de surf',
    'tennis racket': 'raquete de tênis',
    'bottle': 'garrafa',
    'wine glass': 'taça',
    'cup': 'copo',
    'fork': 'garfo',
    'knife': 'faca',
    'spoon': 'colher',
    'bowl': 'tigela',
    'banana': 'banana',
    'apple': 'maçã',
    'sandwich': 'sanduíche',
    'orange': 'laranja',
    'broccoli': 'brócolis',
    'carrot': 'cenoura',
    'hot dog': 'cachorro-quente',
    'pizza': 'pizza',
    'donut': 'rosquinha',
    'cake': 'bolo',
    'chair': 'cadeira',
    'couch': 'sofá',
    'potted plant': 'planta em vaso',
    'bed': 'cama',
    'dining table': 'mesa de jantar',
    'toilet': 'vaso sanitário',
    'tv': 'televisão',
    'laptop': 'notebook',
    'mouse': 'mouse',
    'remote': 'controle remoto',
    'keyboard': 'teclado',
    'cell phone': 'celular',
    'microwave': 'micro-ondas',
    'oven': 'forno',
    'toaster': 'torradeira',
    'sink': 'pia',
    'refrigerator': 'geladeira',
    'book': 'livro',
    'clock': 'relógio',
    'vase': 'vaso',
    'scissors': 'tesoura',
    'teddy bear': 'urso de pelúcia',
    'hair drier': 'secador de cabelo',
    'toothbrush': 'escova de dentes'
}

# Gênero dos objetos (f = feminino, m = masculino)
genero = {
    'person': 'f',
    'bicycle': 'f',
    'car': 'm',
    'motorcycle': 'f',
    'airplane': 'm',
    'bus': 'm',
    'train': 'm',
    'truck': 'm',
    'boat': 'm',
    'traffic light': 'm',
    'fire hydrant': 'm',
    'stop sign': 'f',
    'parking meter': 'm',
    'bench': 'm',
    'bird': 'm',
    'cat': 'm',
    'dog': 'm',
    'horse': 'm',
    'sheep': 'f',
    'cow': 'f',
    'elephant': 'm',
    'bear': 'm',
    'zebra': 'f',
    'giraffe': 'f',
    'backpack': 'm',
    'umbrella': 'm',
    'handbag': 'f',
    'tie': 'f',
    'suitcase': 'f',
    'frisbee': 'm',
    'skis': 'm',
    'snowboard': 'm',
    'sports ball': 'f',
    'kite': 'm',
    'baseball bat': 'm',
    'baseball glove': 'f',
    'skateboard': 'm',
    'surfboard': 'f',
    'tennis racket': 'f',
    'bottle': 'f',
    'wine glass': 'f',
    'cup': 'm',
    'fork': 'f',
    'knife': 'f',
    'spoon': 'f',
    'bowl': 'm',
    'banana': 'f',
    'apple': 'f',
    'sandwich': 'm',
    'orange': 'f',
    'broccoli': 'm',
    'carrot': 'f',
    'hot dog': 'm',
    'pizza': 'f',
    'donut': 'm',
    'cake': 'm',
    'chair': 'f',
    'couch': 'm',
    'potted plant': 'f',
    'bed': 'f',
    'dining table': 'f',
    'toilet': 'm',
    'tv': 'f',
    'laptop': 'm',
    'mouse': 'm',
    'remote': 'm',
    'keyboard': 'f',
    'cell phone': 'm',
    'microwave': 'm',
    'oven': 'm',
    'toaster': 'f',
    'sink': 'f',
    'refrigerator': 'f',
    'book': 'm',
    'clock': 'm',
    'vase': 'm',
    'scissors': 'f',
    'teddy bear': 'm',
    'hair drier': 'm',
    'toothbrush': 'f'
}

CONFIANCA_MINIMA = 0.55

def falar(texto):
    mp3_fp = io.BytesIO()
    tts = gTTS(texto, lang='pt')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def detectar_objetos(cap):
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame da câmera")
        return

    results = model(frame)
    detections = results.pandas().xyxy[0]
    contagem = defaultdict(int)

    for _, row in detections.iterrows():
        if row['confidence'] >= CONFIANCA_MINIMA:
            contagem[row['name']] += 1

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

    print("---> " + frase_final)
    falar(frase_final)

#menu :D
def menu():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            print("\n=== MENU ===")
            print(" digite 1 para Detectar objetos")
            opcao = input("--> ")

            if opcao == "1":
                detectar_objetos(cap)
            else:
                print("Opção inválida, tente novamente!")
    finally:
        cap.release()
        pygame.mixer.quit()

if __name__ == "__main__":
    menu()
