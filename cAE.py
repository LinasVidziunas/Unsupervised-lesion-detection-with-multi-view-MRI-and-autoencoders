from Datapreprocessing.slice import Slice
from plotting import ModelPlotting

import tensorflow as tf
from keras import layers, Model, losses
from keras.layers import Flatten, Dense, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D, LeakyReLU

import numpy as np
from os import listdir, path


def get_data(batch_size: int = 32):
    train_files = listdir("sets/x_train")
    val_files = listdir("sets/x_test")

    x_train = np.zeros((len(train_files), 320, 320))
    x_val = np.zeros((len(val_files), 320, 320))

    train_slices = []
    val_slices = []

    # ValueError thrown when slice does not match the default resolution
    for i, slice_file in enumerate(train_files):
        try:
            _slice = Slice(path.join("sets/x_train", slice_file))
            x_train[i][:][:] = _slice.normalized_pixel_array()
            train_slices.append(_slice)
        except ValueError:
            x_train[i][:][:] = x_train[i - 1][:][:]

    for i, slice_file in enumerate(val_files):
        try:
            _slice = Slice(path.join("sets/x_test", slice_file))
            x_val[i][:][:] = _slice.normalized_pixel_array()
            val_slices.append(_slice)
        except ValueError:
            x_val[i][:][:] = x_val[i - 1][:][:]

    # https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
    # Wrap data in Dataset objects.
    train_data = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    val_data = tf.data.Dataset.from_tensor_slices((x_val, x_val))

    # The batch size must now be set on the Dataset objects.
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    # x_train and x_val and test_slices are temporarily
    # there for the plotting functions to work
    return (train_data, val_data, x_train, x_val, val_slices)


def get_compiled_model():
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
    # autoencoder.summary()
    return autoencoder


strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    autoencoder = get_compiled_model()

BATCH_SIZE_PER_REPLICA = 32
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset, test_dataset, x_train, x_test, test_slices = get_data(
    BATCH_SIZE)

history = autoencoder.fit(
    train_dataset,
    epochs=1,
    validation_data=test_dataset,
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
    [el.reshape(320, 320) for el in reconstructed_images])
