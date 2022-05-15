# A Videogame Controlled with Hand Gestures Recognized by a Neural Network
## PIEDRA, PAPEL SIN TIJERAS

Piedra, Papel sin Tijeras (Rock, Paper without Scissors) was developed as a final project of the Biologically Inspired Complex Systems course at Escuela Técnica Superior de Ingenieros de Telecomunicación, Universitat Politècnica de València.

*You can view the final videopresentation here.*  **ADD LINK TO THE YT VIDEO**

## Neural Network
The deployed neural network was trained using Python. The following files were used for the development:
- Manos.py
- Entrenamiento.py
- Prediccion.py

### Acquiring Photos for Model Training
The photos in the database (which is described in the following chapter) were captured using ***Manos.py***. Detection of hands in the photos was inspired by [this article](https://google.github.io/mediapipe/solutions/hands.html).

<img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/hand_detection.png" width="350" height="280">

In the example a hand is detected. After the detection, the green square is saved as an png file and added to the train/test set, not the whole picture.

### Database of Photos to Train the Model
The following database of photos was used to train the deployed NN model. The model was trained to recognize 2 gestures:
- a palm (300 photos in the train set, 300 photos in the test set)
- a fist (300 photos in the train set, 300 photos in the test set)

[Link to the database](https://drive.google.com/drive/folders/1LASjcZljeLQpKCz3VpGwfoQfgYWynwyj?usp=sharing)

### Training of the Model
After acquiring a sufficient number of photos for both the train teset and the test set, the model was created and trained by using ***Entrenamiento.py***. This piece of code produces two files: 
- Modelo.h5 (the model itself)
- pesos.h5 (the weigths determined for every neuron)

### Gesture Recognition
The trained model is afterwards used to predict which hand gesture is shown by a user. The model reaches a sufficiently low error rate that permits faultless control of the game. For the gesture recongnition, ***Prediccion.py*** is used. In the images below, you can see examples of gesture recognitions.

<p float="">
  <img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/gesture_palm.png" width="350" height="280">
  <img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/gesture_fist.png" width="350" height="280">
</p>
  
## Design of the Game
The game was designed in Unity and is controled with hand gestures recognized by the NN model mentioned above. The design is following:

 <img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/game_design.png" width="800" height="400">

### The goal of the game
The goal of the game is to lead the ball through the playground using hand gestures avoiding hitting blue blocks. When the ball hits a blue block, the game starts again.

### How to control the game
As mentioned above, the game is controlled by two hand gestures: by a palm and by a fist

1. PALM - when a player shows a palm to the camera, the ball goes UP 
2. FIST - when a player shows a fist to the camera, the ball goes DOWN

To see examples of control, see below pictures below.

<p float="">
  <img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/ball_control1.png" width="500" height="280">
  <img src="https://github.com/vavralukas/SCBIO/blob/main/screenshots_readme/ball_control2.png" width="500" height="280">
</p>

## Contributors
- Francesco Bove
- Paola Coves Puelma
- Pablo García San Félix
- Pablo Valero Muñoz
- Lukáš Vávra

