import cv2
import os
import pandas as pd
import numpy as np

# Caminhos (ajuste para as suas pastas)
dir_320 = 'tmp/yolov8-seg'
dir_1024 = 'tmp/yolo-1024-95pbt-5pbv'
dir_provas = 'images-teste-comparacao2'
os.makedirs(dir_provas, exist_ok=True)

lista_provas = []

# Pega a lista de arquivos de uma das pastas
arquivos = sorted([f for f in os.listdir(dir_320) if f.endswith(('_raw_debug.png'))])

print(f"Gerando {len(arquivos)} provas visuais...")

for nome in arquivos:
    img320 = cv2.imread(os.path.join(dir_320, nome))
    img1024 = cv2.imread(os.path.join(dir_1024, nome))

    if img1024 is None:
        print(f"Aviso: {nome} não encontrado na pasta 1024. Pulando...")
        continue

    # Cria uma linha divisória preta de 10 pixels entre as imagens
    altura, largura, _ = img320.shape
    divisor = np.zeros((altura, 10, 3), dtype=np.uint8)

    # Concatena as imagens horizontalmente (320px | Divisor | 1024px)
    prova = np.hstack((img320, divisor, img1024))

    # Salva a imagem final
    caminho_save = os.path.join(dir_provas, f"PROVA_{nome}")
    cv2.imwrite(caminho_save, prova)

    # Adiciona ao inventário do CSV
    lista_provas.append({
        'Arquivo': nome,
        'TP_v8Seg': 0, 'FP_v8Seg': 0, 'FN_v8Seg': 0,
        'TP_1024': 0, 'FP_1024': 0, 'FN_1024': 0
    })

# Gera o CSV para preenchimento manual
df = pd.DataFrame(lista_provas)
df.to_csv('matriz_confusao_manual2.csv', index=False)
print("Concluído! Abra o CSV e a pasta de provas para começar a análise.")