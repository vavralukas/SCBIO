import cv2
import mediapipe as mp
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import pyautogui
from keras_preprocessing.image import img_to_array

modelo = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/Modelo.h5"
peso = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/pesos.h5"
cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/Fotos/validacion"
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

cap = cv2.VideoCapture(0)

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()

dibujo = mp.solutions.drawing_utils

while (1):
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
                pto_i1 = posiciones[3]
                pto_i2 = posiciones[17]
                pto_i3 = posiciones[10]
                pto_i4 = posiciones[0]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100)
                ancho, alto = (x1+200), (y1+200)
                x2, y2 = x1+ancho, y1+alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = tf.keras.preprocessing.image.img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)

                if respuesta == 0:
                    print(resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "{}".format(dire_img[0]), (x1, y1-5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                elif respuesta == 1:
                    print(resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "{}".format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                    #pyautogui.press("volumeup")
                elif respuesta == 2:
                    print(resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, "{}".format(dire_img[2]), (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                    #pyautogui.press("volumedown")
                else:
                    cv2.putText(frame, "letra desconocida", (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()