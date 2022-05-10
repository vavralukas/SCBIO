# LIBRERIAS
import cv2                  # libreria de Opencv-python
import os                   # movernos entre carpetas
import mediapipe as mp      # Detector de manos

# Creamos carpetas para almacenar las fotos de entrenamiento y validacion
nombre = "L"
directorio = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/Fotos/validacion"
carpeta = directorio + "/" + nombre

# por si no esta creada la carpeta que se cree sola
if not os.path.exists(carpeta):
    print("Carpeta creada:", carpeta)
    os.makedirs(carpeta)

# contador para el nombre de las fotos
cont = 0

# Iniciamos la camara (0 para WebCam 1 para externa)
cap = cv2.VideoCapture(0)

# Objeto que almacena deteccion y seguimiento de las manos

clase_manos = mp.solutions.hands            # detector de manos de Google
manos = clase_manos.Hands()

# dibuja los puntos cardinales de la manos
dibujo = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()                                     # lectura de camara
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # Opencv entrega los colores en BGR y queremos RGB
    copia = frame.copy()                                        # copia para procesamiento posterior
    resultado = manos.process(color)
    posiciones = []                                             # coordenadas de los puntos

    if resultado.multi_hand_landmarks:                          # si detecto alguna mano
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i1 = posiciones[4]
                pto_i2 = posiciones[20]
                pto_i3 = posiciones[12]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100)
                ancho, alto = (x1+200), (y1+200)
                x2, y2 = x1+ancho, y1+alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation= cv2.INTER_CUBIC)
            cv2.imwrite(carpeta+"/L_{}.jpg".format(cont), dedos_reg)
            cont = cont + 1

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:       # Cuando tengamos guardadas 300 fotos
        break

cap.release()
cv2.destroyAllWindows()
