# import keras
from keras import layers, Model
# from keras.datasets import mnist
import numpy as np
from keras.layers import Flatten, Dense, Reshape
from matplotlib import pyplot as plt
from os import listdir, path

from Dataprocessing.patient_data_preprocessing import Slice

train_slices = listdir("sets/x_train")
test_slices = listdir("sets/x_test")

x_train = np.zeros((len(train_slices), 320, 320))
x_test = np.zeros((len(test_slices), 320, 320))

sagital_slices = []

for i, s in enumerate(train_slices):
    try:
        x_train[i][:][:] = Slice(path.join("sets/x_train",
                                           s)).normalized_pixel_array()
    except:
        x_train[i][:][:] = x_train[i - 1][:][:]

for i, slice in enumerate(test_slices):
    try:
        x_test[i][:][:] = Slice(path.join("sets/x_test",
                                          s)).normalized_pixel_array()
    except:
        x_test[i][:][:] = x_test[i - 1][:][:]

input = layers.Input(shape=(320, 320, 1))

# Encoder
x = layers.Conv2D(120, (12, 12), activation="relu", padding="same")(input)
x = layers.Conv2D(120, (12, 12), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(140, (7, 7), activation="relu", padding="same")(x)
x = layers.Conv2D(140, (7, 7), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(160, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(160, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(180, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(180, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)

x = layers.Conv2DTranspose(200, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(180, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(180, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2D(200, (3, 3), activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(160, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2D(160, (7, 7), activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(140, (5, 5),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2D(140, (12, 12), activation="relu", padding="same")(x)
x = layers.Conv2D(140, (12, 12), activation="sigmoid", padding="same")(x)
decoded = layers.Conv2D(100, (7, 7), activation="relu", padding="same")(x)

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

autoencoder.fit(
    x_train,
    x_train,
    epochs=1000,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test),
)

decoded_images = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(320, 320))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_images[i].reshape(320, 320))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig("bigsave.png")
