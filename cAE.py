import keras
from keras import layers, Model
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

input = layers.Input(shape=(320, 320, 1))

# Encoder
x = layers.Conv2D(100, (7, 7), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((4, 4), padding="same")(x)
x = layers.Conv2D(50, (5, 5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((4, 4), padding="same")(x)
x = layers.Conv2D(25, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(10, (3, 3), activation="relu", padding="same")(x)




# Decoder
x = layers.Conv2DTranspose(25, (3, 3), strides=4, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(50, (3, 3), strides=4, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


x_train =
x_test =


autoencoder.fit(x_train, x_train,
    epochs=30,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
)