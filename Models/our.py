from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Reshape, UpSampling2D, Dense, LeakyReLU


def ourBestModel():
    input = Input(shape=(320, 320, 1))

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

    return Model(input, decoded)
