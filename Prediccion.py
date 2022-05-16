import cv2
import mediapipe as mp
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import pyautogui
from keras_preprocessing.image import img_to_array

# VARIABLES
flag = 1                            # used to initiate the game
gesture_names = ['Palma','Puno']    # names of hand gestures

# LOADING MODEL
modelo = "Modelo.h5"
peso = "pesos.h5"
cnn = load_model(modelo)
cnn.load_weights(peso)

# INITIATE CAMERA
cap = cv2.VideoCapture(0)

# CREATE OBJECT WHICH DETECTS HANDS AND THEIR MOVEMENTS
clase_manos = mp.solutions.hands            
manos = clase_manos.Hands()

# SHOW CARDINAL POINTS OF DETECTED HANDS
dibujo = mp.solutions.drawing_utils

# MAIN LOOP 
while True:
    ret, frame = cap.read()                                     # read frames from the camera
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # convert BGR frame to RGB frame
    copia = frame.copy()                                        # copy of the frame for future processing
    resultado = manos.process(color)                            # final frame from camera to be processed
    posiciones = []                                             # array of hand mark coordinates necessary for image processing

    # IF CONDITION - HAND IS DETECTED?
    if resultado.multi_hand_landmarks:      

        # LOOP FOR EVERY HAND DETECTED                    
        for mano in resultado.multi_hand_landmarks:

            # SAVE AND DRAW HAND MARK COORDINATES 
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto)
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS) # display conncetions between marks 

            # IF HAND MARK COORDINATES WERE SAVED
            if len(posiciones) != 0:
                # coordinates used for further image processing
                pto_central = posiciones[9]                     # mark in the centre of the hand
                pto_5 = posiciones[5]                           # auxiliary variable to determine the size of the final frame
                pto_17 = posiciones[17]                         # auxiliary variable to determine the size of the final frame

                # determining the size/coordinates of the final frame which is saved
                distancia = int(np.round_(np.sqrt((pto_5[1] - pto_17[1])**2 + (pto_5[2] - pto_17[2])**2)))
                x1, y1 = (pto_central[1]-int(1.5*distancia)), (pto_central[2]-int(1.8*distancia))
                x2, y2 = (x1+3*distancia), (y1+int(3.5*distancia))    

                # conditions to prevent errors when frame leaves camera
                if y1 < 0:                                       
                    y1 = 0
                if x1 < 0:
                    x1 = 0

                # cut final frame according to determined coordinates
                final_frame = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # resize frame    
                final_frame = cv2.resize(final_frame, (3*distancia, int(3.5*distancia)), interpolation= cv2.INTER_CUBIC)
                final_frame = cv2.resize(final_frame, (200, 200))

                # predict hand gesture
                x = tf.keras.preprocessing.image.img_to_array(final_frame)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)

                # show prediction (as a frame color and title around a hand gesture)
                # send signal to the game - UP or DOWN
                if respuesta == 0:
                    # frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, "{}".format(gesture_names[0]), (x1, y1-5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                    # signal UP
                    pyautogui.press("up")

                else: # respuesta == 1:
                    # frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "{}".format(gesture_names[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                    # signal DOWN
                    pyautogui.press("down")

                # start the game 
                if flag == 1:
                    os.startfile("Pelotita.exe")
                    flag = 0


    # DISPLAY CAMERA 
    cv2.imshow("Video", frame)
    cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST,1)

    # TERMINATE LOOP
    k = cv2.waitKey(1)          
    if k == 27 :       # terminate loop if ESC is pressed
        break

# CLOSE CAMERA
cap.release()
cv2.destroyAllWindows()
