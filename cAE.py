from Datapreprocessing.slice import Slice

from keras import layers, Model, losses
from keras.layers import Flatten, Dense, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D
from matplotlib import pyplot as plt
import numpy as np

from os import listdir, path

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

for i, s in enumerate(test_slices):
    try:
        x_test[i][:][:] = Slice(path.join("sets/x_test",
                                          s)).normalized_pixel_array()
    except:
        x_test[i][:][:] = x_test[i - 1][:][:]

input = layers.Input(shape=(320, 320, 1))

# Encoder
conv1 = Conv2D(32, (7, 7), activation='relu',
               padding='same')(input)  # 320 x 320 x 32
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
conv2 = Conv2D(64, (5, 5), activation='relu',
               padding='same')(pool1)  # 160 x 160 x 64
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(
    pool2)  # 80 x 80 x 128 (small and thick)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 7 x 7 x 64

conv6 = Conv2D(32, (3, 3), activation='relu',
               padding='same')(pool3)  # 160 x 160 x 32
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
up2 = UpSampling2D((2, 2))(conv6)  # 28 x 28 x 64
conv7 = Conv2D(32, (3, 3), activation='relu',
               padding='same')(up2)  # 160 x 160 x 32
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
up3 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64
conv8 = Conv2D(32, (5, 5), activation='relu',
               padding='same')(up3)  # 160 x 160 x 32
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv8)
conv8 = BatchNormalization()(conv8)
up4 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64
conv9 = Conv2D(32, (5, 5), activation='relu',
               padding='same')(up4)  # 160 x 160 x 32
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv9)
conv9 = BatchNormalization()(conv9)
decoded = Conv2D(1, (7, 7), activation='sigmoid',
                 padding='same')(conv9)  # 320 x 320 x 1

# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mae"])
autoencoder.summary()

autoencoder.fit(
    x_train,
    x_train,
    epochs=15,
    batch_size=32,
    validation_data=(x_test, x_test),
)

test_abnormal = []
test_normal = []

for a in test_slices:
    if Slice(path.join("sets/x_test", a)).get_abnormality() is True:
        test_abnormal.append(Slice(path.join("sets/x_test", a)).normalized_pixel_array())
    elif Slice(path.join("sets/x_test", a)).get_abnormality() is False:
        test_normal.append(Slice(path.join("sets/x_test", a)).normalized_pixel_array())


decoded_normal = autoencoder.predict(test_normal)
loss_normal = losses.mae(decoded_normal, test_normal)

plt.hist(loss_normal[None, :], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of images")
plt.savefig("figure_normal.png")

decoded_abnormal = autoencoder.pedict(test_abnormal)
loss_abnormal = loss_normal.mae(decoded_abnormal, test_abnormal)
plt.hist(loss_abnormal[None, :], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of images")
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
