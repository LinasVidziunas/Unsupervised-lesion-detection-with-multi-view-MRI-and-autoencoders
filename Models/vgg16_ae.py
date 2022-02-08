from keras import Model
from keras.layers import Input, Conv2D, Reshape
from keras.layers import UpSampling2D, Concatenate
from keras.applications.vgg16 import VGG16


# Solution for 3 channel
# https://stackoverflow.com/questions/52065412/how-to-use-1-channel-images-as-inputs-to-a-vgg-model
# Might be smart to rewrite as one channel
def vgg16(input_size=(320, 320, 1)):
    """An autoencoder implementation of VGG16 architecture"""

    inp = Input(shape=input_size)

    mg_conc = Concatenate()([inp, inp, inp])

    encoded = VGG16(input_tensor=mg_conc, include_top=False, weights=None)(inp)

    ################### latent ###################
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='latent')(encoded)
    x = UpSampling2D((2, 2))(x)

    ################### decoder ###################

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # originally as the line under but changed to return one channel
    # decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    return Model(inp, decoded)


def vgg16_dense(input_size=(320, 320, 1), dense_size: int = 160):
    """An autoencoder implementation of VGG16 architecture with a dense layer as the bottlenek"""

    inp = Input(shape=input_size)

    mg_conc = Concatenate()([inp, inp, inp])

    encoded = VGG16(input_tensor=mg_conc, include_top=False, weights=None,
                    classes=dense_size)(inp)

    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='latent')(encoded)
    x = UpSampling2D((2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # originally as the line under but changed to return one channel
    # decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    return Model(inp, decoded)
