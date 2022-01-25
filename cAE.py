# import keras
from keras import layers, Model
# from keras.datasets import mnist
# import numpy as np
from keras.layers import Flatten, Dense, Reshape
# from matplotlib import pyplot as plt

input = layers.Input(shape=(320, 320, 1))

# Encoder
x = layers.Conv2D(50, (9, 9), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(40, (7, 7), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(30, (5, 5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(15, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = Flatten()(x)
encoded = Dense(400, activation='softmax')(x)

# DECODER
x = Reshape((20, 20, 1))(encoded)
x = layers.Conv2DTranspose(15, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(30, (4, 4),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(40, (5, 5),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(50, (7, 7),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
decoded = layers.Conv2D(1, (9, 9), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# Det vil si at x_train = axial_set[0](som skal inneholde "training"),
# og x_test = [2](som skal inneholdet "testing"). Derfor må det først lagres som en streng.

# Må også normalize data før setter inn.

# x_train =
# x_test =

# autoencoder.fit(x_train, x_train,
#     epochs=15,
#     batch_size=128,
#     shuffle=True,
#     validation_data=(x_test, x_test),
# )
#
# decoded_images = autoencoder.predict(x_test)
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n + 1):
#     # Display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(320, 320))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_images[i].reshape(320, 320))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()import keras

input = layers.Input(shape=(320, 320, 1))

# Encoder
x = layers.Conv2D(50, (9, 9), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(40, (7, 7), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(30, (5, 5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(15, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = Flatten()(x)
encoded = Dense(400, activation='softmax')(x)

# DECODER
x = Reshape((20, 20, 1))(encoded)
x = layers.Conv2DTranspose(15, (3, 3),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(30, (4, 4),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(40, (5, 5),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
x = layers.Conv2DTranspose(50, (7, 7),
                           strides=2,
                           activation="relu",
                           padding="same")(x)
decoded = layers.Conv2D(1, (9, 9), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# Det vil si at x_train = axial_set[0](som skal inneholde "training"),
# og x_test = [2](som skal inneholdet "testing"). Derfor m det frst lagres som en streng.

# M ogs normalize data fr setter inn.

# x_train =
# x_test =

# autoencoder.fit(x_train, x_train,
#     epochs=15,
#     batch_size=128,
#     shuffle=True,
#     validation_data=(x_test, x_test),
# )
#
# decoded_images = autoencoder.predict(x_test)
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n + 1):
#     # Display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(320, 320))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_images[i].reshape(320, 320))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
