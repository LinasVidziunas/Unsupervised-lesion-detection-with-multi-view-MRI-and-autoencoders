from Datapreprocessing.slice import Slice
from plotting import ModelPlotting

from keras import layers, Model, losses
from keras.layers import Flatten, Dense, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D, LeakyReLU

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
x = Conv2D(64, (7, 7), activation='relu', padding='same')(input)
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

x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
x = Flatten()(x)
# units = x.shape[1]

# bottleneck

x = Dense(1600)(x)
x = LeakyReLU(alpha=0.2)(x)

# Decoder

x = Reshape((40, 40, 1))(x)

x = Conv2D(512, (5, 5), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)


# Autoencoder
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["mae"])
autoencoder.summary()

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=1000,
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

# Plotting the MSE distrubution of normal slices
decoded_normal = autoencoder.predict(test_normal)
loss_normal = losses.mae(decoded_normal.reshape(len(test_normal), 320 * 320),
                         test_normal.reshape(len(test_normal), 320 * 320))

decoded_abnormal = autoencoder.predict(test_abnormal)
loss_abnormal = losses.mae(
    decoded_abnormal.reshape(len(test_abnormal), 320 * 320),
    test_abnormal.reshape(len(test_abnormal), 320 * 320))

plot = ModelPlotting(history, save_in_dir="sets")

plot.plot_mae_train_vs_val()
plot.plot_loss_train_vs_val()

plot.histogram_mae_loss(loss_normal, loss_abnormal)
plot.histogram_mae_loss_seperate(loss_normal, loss_abnormal)

reconstructed_images = autoencoder.predict(x_test)

plot.input_vs_reconstructed_images(
    [el.reshape(320, 320) for el in x_test],
    [el.reshape(320, 320) for el in reconstructed_images]
)
