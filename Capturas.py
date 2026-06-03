import cv2
import os

detector_rostro = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


NOMBRE_PERSONA = input("Nombre de la persona a capturar: ")
CARPETA_BASE   = "dataset"
FOTOS_A_TOMAR  = 20      

carpeta = os.path.join(CARPETA_BASE, NOMBRE_PERSONA)
os.makedirs(carpeta, exist_ok=True)

cap = cv2.VideoCapture(0)
contador = 0

print(f"Capturando fotos de '{NOMBRE_PERSONA}'. Mira la cámara.")
print("Presiona ESPACIO para capturar, ESC para salir.")

while contador < FOTOS_A_TOMAR:
    ret, frame = cap.read()
    if not ret:
        break

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray   = cv2.equalizeHist(gray)
    rostros = detector_rostro.detectMultiScale(gray, 1.15, 6, minSize=(80, 80))

    vista = frame.copy()

    for (x, y, w, h) in rostros:
        cv2.rectangle(vista, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(vista, f"Fotos: {contador}/{FOTOS_A_TOMAR}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vista, "ESPACIO=capturar  ESC=salir", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Capturar fotos", vista)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:  
        break

    if tecla == 32 and len(rostros) > 0:   # espacio
        (x, y, w, h) = rostros[0]
        # Guarda SOLO el recorte del rostro en escala de grises
        rostro_recortado = gray[y:y+h, x:x+w]
        rostro_recortado = cv2.resize(rostro_recortado, (100, 100))
        ruta = os.path.join(carpeta, f"{contador:03d}.jpg")
        cv2.imwrite(ruta, rostro_recortado)
        print(f"  Guardada: {ruta}")
        contador += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nListo. {contador} fotos guardadas en '{carpeta}/'")
