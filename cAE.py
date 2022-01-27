from Datapreprocessing.slice import Slice

from keras import layers, Model, losses
from keras.layers import Flatten, Dense, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, path


train_files = listdir("sets/x_train")
test_files = listdir("sets/x_test")

x_train = np.zeros((len(train_files), 320, 320))
x_test = np.zeros((len(test_files), 320, 320))

train_slices = []
test_slices = []

# ValueError thrown when slice does not match the default resolution
for i, slice_file in enumerate(train_files):
    try:
        _slice = Slice(path.join("sets/x_train", slice_file))
        x_train[i][:][:] = _slice.normalized_pixel_array()
        train_slices.append(_slice)
    except ValueError:
        x_train[i][:][:] = x_train[i - 1][:][:]

for i, slice_file in enumerate(test_files):
    try:
        _slice = Slice(path.join("sets/x_test", slice_file))
        x_test[i][:][:] = _slice.normalized_pixel_array()
        test_slices.append(_slice)
    except ValueError:
        x_test[i][:][:] = x_test[i - 1][:][:]

input = layers.Input(shape=(320, 320, 1))

# Encoder
x = Conv2D(64, (5, 5), activation='relu', padding='same')(input)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(512, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Alternative 1: latent space (spatial)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)

# Alternative 2: latent space (dense) ->
# In order to use this must remove one up-sampling layer from the decoder.

# x = Flatten()(x)
# x = Dense(1600, activation='softmax')(x)
# DECODER
# d = Reshape((40, 40, 1))(x)

# decoder
x = Conv2D(512, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["mae"])
autoencoder.summary()

autoencoder.fit(
    x_train,
    x_train,
    epochs=150,
    batch_size=32,
    validation_data=(x_test, x_test),
)

test_abnormal_l = []
test_normal_l = []

for _slice in test_slices:
    if _slice.get_abnormality() == 1:
        test_abnormal_l.append(_slice)
    elif _slice.get_abnormality() == 0:
        test_normal_l.append(_slice)

test_abnormal = np.zeros((len(test_abnormal_l), 320, 320))
test_normal = np.zeros((len(test_normal_l), 320, 320))

for i, _slice in enumerate(test_abnormal_l):
    test_abnormal[i][:][:] = _slice.normalized_pixel_array()

for i, _slice in enumerate(test_normal_l):
    test_normal[i][:][:] = _slice.normalized_pixel_array()

# plotting the MSE distrubution of normal slices
decoded_normal = autoencoder.predict(test_normal)
loss_normal = losses.mae(decoded_normal.reshape(
    len(test_normal), 320 * 320), test_normal.reshape(len(test_normal), 320*320))
plt.hist(loss_normal[None, :], bins=100)
plt.xlabel("Train loss")
plt.ylabel("No. of images normal")
plt.savefig("figure_abnormal.png")

# plotting the MSE distribution of abnormal slices
decoded_abnormal = autoencoder.predict(test_abnormal)
loss_abnormal = losses.mae(decoded_abnormal.reshape(len(
    test_abnormal), 320 * 320), test_abnormal.reshape(len(test_abnormal), 320*320))
plt.hist(loss_abnormal[None, :], bins=100)
plt.xlabel("Train loss")
plt.ylabel("No. of images abnormal")
plt.savefig("figure_abnormal.png")

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

plt.savefig("figure_bigsave.png")
