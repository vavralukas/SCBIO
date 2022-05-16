# LIBRARIES
import cv2                  
import os                   
import mediapipe as mp
import numpy as np
import time


# CREATE PATH TO FOLDERS WHERE PHOTOS WILL BE SAVED
nombre = "palm"
directorio = "C:/Users/lukas/Documents/SCBIO/training_set"
carpeta = directorio + "/" + nombre

# CREATE FOLDER IF IT DOES NOT EXIST YET
if not os.path.exists(carpeta):
    print("Carpeta creada:", carpeta)
    os.makedirs(carpeta)

# LOOP COUNTER (also used as an input where creating photos' names)
cont = 1

# INITIATE CAMERA
cap = cv2.VideoCapture(0)

# CREATE OBJECT WHICH DETECTS HANDS AND THEIR MOVEMENTS
clase_manos = mp.solutions.hands            
manos = clase_manos.Hands()

# SHOW CARDINAL POINTS OF DETECTED HANDS
dibujo = mp.solutions.drawing_utils

# MAIN LOOP - TAKING IMAGES OF HANDS
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

            # resize and save frame as jpg to elected folder     
            final_frame = cv2.resize(final_frame, (3*distancia, int(3.5*distancia)), interpolation= cv2.INTER_CUBIC)
            final_frame = cv2.resize(final_frame, (200, 200))
            cv2.imwrite(carpeta+"/"+nombre+"_{}.jpg".format(cont), final_frame)

            # increase counter
            cont = cont + 1
            time.sleep(0.05)

    # DISPLAY CAMERA 
    cv2.imshow("Video", frame)

    # TERMINATE LOOP
    k = cv2.waitKey(1)          
    if k == 27 or cont > 100:       # terminate loop if ESC is pressed or 100 photos have been taken
        break

# CLOSE CAMERA
cap.release()
cv2.destroyAllWindows()
