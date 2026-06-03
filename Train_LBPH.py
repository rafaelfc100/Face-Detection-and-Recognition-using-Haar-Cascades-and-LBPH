import cv2
import os
import numpy as np

CARPETA_BASE = "dataset"

imagenes  = []   # píxeles de cada rostro
etiquetas = []   # número de ID por persona
nombres   = {}   # diccionario id → nombre

print("Leyendo dataset...")

personas = sorted(os.listdir(CARPETA_BASE))

for id_persona, nombre in enumerate(personas):
    carpeta = os.path.join(CARPETA_BASE, nombre)
    if not os.path.isdir(carpeta):
        continue

    nombres[id_persona] = nombre
    fotos = [f for f in os.listdir(carpeta) if f.endswith(".jpg")]

    for foto in fotos:
        ruta  = os.path.join(carpeta, foto)
        img   = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        imagenes.append(img)
        etiquetas.append(id_persona)

    print(f"  {nombre}: {len(fotos)} fotos  (ID={id_persona})")

if not imagenes:
    print("No se encontraron imágenes. Ejecuta primero paso1_capturar.py")
    exit()


print("\nEntrenando modelo LBPH...")
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.train(imagenes, np.array(etiquetas))
reconocedor.save("modelo.xml")

with open("etiquetas.txt", "w") as f:
    for id_, nom in nombres.items():
        f.write(f"{id_},{nom}\n")

print("Modelo guardado en: modelo.xml")
print("Etiquetas guardadas en: etiquetas.txt")
print(f"Personas reconocibles: {list(nombres.values())}")
print("\nEjecuta ahora: python paso3_reconocer.py")