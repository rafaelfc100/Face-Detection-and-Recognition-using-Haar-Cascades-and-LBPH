import cv2
import numpy as np
import os

# ── Todos los clasificadores disponibles ─────────────────────────
CLASIFICADORES = {
    "1": ("Rostro Frontal",    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    "2": ("Rostro Perfil",     cv2.data.haarcascades + "haarcascade_profileface.xml"),
    "3": ("Cuerpo Superior",   cv2.data.haarcascades + "haarcascade_upperbody.xml"),
    "4": ("Cuerpo Completo",   cv2.data.haarcascades + "haarcascade_fullbody.xml"),
    "5": ("Ojos",              cv2.data.haarcascades + "haarcascade_eye.xml"),
    "6": ("Ojos con anteojos", cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"),
    "7": ("Sonrisa / Boca",    cv2.data.haarcascades + "haarcascade_smile.xml"),
}

COLORES = {
    "1": (255, 200,   0),   # cian       - rostro frontal
    "2": (255,   0, 200),   # magenta    - perfil
    "3": (  0, 255, 100),   # verde      - cuerpo superior
    "4": (  0, 200, 255),   # amarillo   - cuerpo completo
    "5": (  0, 120, 255),   # naranja    - ojos
    "6": (100, 255, 255),   # amarillo c - ojos con anteojos
    "7": (180,   0, 255),   # violeta    - sonrisa
}
PARAMS = {
    "1": dict(scaleFactor=1.15, minNeighbors=6, minSize=(50, 50)),
    "2": dict(scaleFactor=1.15, minNeighbors=6, minSize=(50, 50)),
    "3": dict(scaleFactor=1.15, minNeighbors=6, minSize=(50, 50)),
    "4": dict(scaleFactor=1.10, minNeighbors=4, minSize=(60,150)),
    "5": dict(scaleFactor=1.10, minNeighbors=8, minSize=(20, 20)),
    "6": dict(scaleFactor=1.10, minNeighbors=8, minSize=(20, 20)),
    "7": dict(scaleFactor=1.20, minNeighbors=20, minSize=(30, 30)),
}

# interfaz
def mostrar_menu():
    print("Clasificadores disponibles:")
    for key, (nombre, _) in CLASIFICADORES.items():
        print(f"  [{key}] {nombre}")
    print("1,2,3,4,5,6, y asi, los que se van a detectar")

def elegir_detectores():
    mostrar_menu()
    while True:
        entrada = input("cuales -> ").strip()
        claves = [c.strip() for c in entrada.split(",")]
        validas = [c for c in claves if c in CLASIFICADORES]
        if validas:
            print("\ndetectores:")
            for c in validas:
                print(f"good")
            return validas
        print("de nuez")

def usar_reconocimiento():
    if not os.path.exists("modelo.xml") or not os.path.exists("etiquetas.txt"):
        return False
    resp = input("\n usar modelo entrenado?[s/n]: ").strip().lower()
    return resp == "s"

def cargar_modelo():
    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.read("modelo.xml")
    nombres = {}
    with open("etiquetas.txt", "r") as f:
        for linea in f:
            id_, nombre = linea.strip().split(",")
            nombres[int(id_)] = nombre
    return reconocedor, nombres

def detectar(frame, gray, detectores_activos, reconocedor=None, nombres=None, umbral=70):
    salida = frame.copy()
    conteos = {}

    rostros_encontrados = []

    for clave in detectores_activos:
        nombre_det, ruta = CLASIFICADORES[clave]
        detector = cv2.CascadeClassifier(ruta)
        color = COLORES[clave]

        # Ojos y boca: buscar DENTRO del rostro frontal si fue detectado
        if clave in ("5", "6", "7") and rostros_encontrados and "1" in detectores_activos:
            total = 0
            for (fx, fy, fw, fh) in rostros_encontrados:
                # Ojos: mitad superior; Boca: tercio inferior
                if clave in ("5", "6"):
                    roi = gray[fy : fy + fh//2, fx : fx+fw]
                    oy, ox = fy, fx
                else:   # sonrisa
                    roi = gray[fy + fh//2 : fy+fh, fx : fx+fw]
                    oy, ox = fy + fh//2, fx

                detecciones = detector.detectMultiScale(roi, **PARAMS[clave])
                for (ex, ey, ew, eh) in detecciones:
                    ax, ay = ox+ex, oy+ey
                    cv2.rectangle(salida, (ax, ay), (ax+ew, ay+eh), color, 1)
                    cv2.putText(salida, nombre_det, (ax, ay-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                    total += 1
            conteos[nombre_det] = total
            continue
        detecciones = detector.detectMultiScale(gray, **PARAMS[clave])
        conteos[nombre_det] = len(detecciones)

        for (x, y, w, h) in detecciones:
            cv2.rectangle(salida, (x, y), (x+w, y+h), color, 2)

            if clave == "1" and reconocedor is not None:
                roi_pred = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                id_pred, distancia = reconocedor.predict(roi_pred)
                if distancia < umbral:
                    etiqueta = f"{nombres.get(id_pred, '?')} ({distancia:.0f})"
                else:
                    etiqueta = f"Desconocido ({distancia:.0f})"
            else:
                etiqueta = nombre_det

            cv2.putText(salida, etiqueta, (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if clave == "1":
                rostros_encontrados.append((x, y, w, h))

    y_hud = 25
    for nombre_det, cnt in conteos.items():
        cv2.putText(salida, f"{nombre_det}: {cnt}", (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y_hud += 20

    cv2.putText(salida, "ESC = salir", (10, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return salida

def main():
    claves_activas = elegir_detectores()

    reconocedor, nombres = None, None
    if "1" in claves_activas and usar_reconocimiento():
        reconocedor, nombres = cargar_modelo()
        print("Reconocimiento activado.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo acceder a la cámara.")
        return

    print("\nESC para salir y click en la venatana porque se traba\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        salida = detectar(frame, gray, claves_activas, reconocedor, nombres)
        cv2.imshow("Detector en Tiempo Real", salida)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()