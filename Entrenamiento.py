import tensorflow.python.keras.optimizers
import tensorflow as tf
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.callbacks import TensorBoard

k.clear_session()

# FOLDERS WITH TRAIN SET AND VALIDATION SET
datos_entrenamiento = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/Fotos/entrenamiento"
datos_validacion = "C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp/Fotos/validacion"

# PARAMETERS OF TRAINING
iteraciones = 200               # number of training iterations
altura, longitud = 200, 200     # image size
batch_size = 1
pasos = 300/1                   # size of train set
pasos_validacion = 300/1        # size of validation set
clases = 2                       # number of classes model should recognize   
lr = 0.0005                      # learning rate

# parameters of filters
filtrosconv1 = 32
filtrosconv2 = 64
filtrosconv3 = 128
tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)
tam_filtro3 = (2, 2)
tam_pool = (2, 2)


# PREPROCESSING 

# train set - step 1
preprocesamiento_entre = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True,
)

# validation set - step 1
preprocesamiento_vali = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
)

# train set - step 2
imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical",
)

# validation set - step 2
imagen_validacion = preprocesamiento_entre.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode="categorical",
)

# CREATING NEURAL NETWORK
cnn = tf.keras.models.Sequential()

# adding convolutional and pooling layers
cnn.add(tf.keras.layers.Convolution2D(filtrosconv1, tam_filtro1, padding="same", input_shape=(altura, longitud, 3), activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=tam_pool))
cnn.add(tf.keras.layers.Convolution2D(filtrosconv2, tam_filtro2, activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=tam_pool))
cnn.add(tf.keras.layers.Convolution2D(filtrosconv3, tam_filtro3, activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=tam_pool))

# additional layers
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(640, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(clases, activation="softmax"))

# optimizing model
optimizar = tf.keras.optimizers.Adam(learning_rate=lr)
cnn.compile(loss = "categorical_crossentropy", optimizer=optimizar, metrics=["accuracy"])
BoardCNN2 = TensorBoard(log_dir="C:/Users/16pao/OneDrive/Escritorio/Teleco/Scbio/Flappymp")

# TRAINING MODEL
cnn.fit(imagen_entreno, steps_per_epoch=pasos, callbacks=BoardCNN2, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos)

# SAVE TRAINED MODEL AND WEIGHTS
cnn.save("Modelo.h5")
cnn.save_weights("pesos.h5")
