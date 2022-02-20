from keras import Model
from keras.layers import Input, Conv2D, Reshape, Flatten, Dense
from keras.layers import UpSampling2D, Concatenate, MaxPool2D
from keras.layers import BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16


# Solution for 3 channel
# https://stackoverflow.com/questions/52065412/how-to-use-1-channel-images-as-inputs-to-a-vgg-model
# Might be smart to rewrite as one channel
def vgg16(input_size=(320, 320, 1)):
    """An autoencoder implementation of VGG16 architecture"""

    inp = Input(shape=input_size)

    mg_conc = Concatenate()([inp, inp, inp])

    encoded = VGG16(input_tensor=mg_conc, include_top=False, weights=None)

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

    encoded = VGG16(input_tensor=mg_conc, include_top=True, weights=None,
                    classes=dense_size)

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


def own_vgg16_conv2d_block(previous_layer, filters, batchNorm: bool):
    x = Conv2D(filters=filters, kernel_size=(3, 3),
               padding="same", activation="relu")(previous_layer)
    if batchNorm:
        x = BatchNormalization()(x)

    return x


def own_vgg16_encoder_block(previous_layer, filters: int,
                            conv2d_layers: int = 2,
                            batchNorm: bool = False,
                            dropout_rate: int = 0):

    block = own_vgg16_conv2d_block(previous_layer, filters, batchNorm)

    for _ in range(conv2d_layers-1):
        block = own_vgg16_conv2d_block(block, filters, batchNorm)

    if dropout_rate != 0:
        block = Dropout(dropout_rate)

    block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block)

    return block


def own_vgg16_decoder_block(previous_layer, filters: int,
                            conv2d_layers: int = 2,
                            batchNorm: bool = False,
                            dropout_rate: int = 0):
    block = None

    for i in range(conv2d_layers):
        block = own_vgg16_conv2d_block(previous_layer, filters, batchNorm)

    if dropout_rate != 0:
        block = Dropout(dropout_rate)(block)

    block = UpSampling2D(size=(2, 2))(block)

    return block


def own_vgg16(inputs, input_shape=(384, 384, 1), dense_size: int = 80):
    encoder_filters = [64, 128, 256, 512, 512]
    decoder_filters = [512, 512, 256, 128, 64]

    # inputs = Input(shape=input_shape)

    b1 = own_vgg16_encoder_block(
        previous_layer=inputs, filters=encoder_filters[0], conv2d_layers=2, batchNorm=True)
    b2 = own_vgg16_encoder_block(
        previous_layer=b1, filters=encoder_filters[1], conv2d_layers=2, batchNorm=True)
    b3 = own_vgg16_encoder_block(
        previous_layer=b2, filters=encoder_filters[2], conv2d_layers=3, batchNorm=True)
    b4 = own_vgg16_encoder_block(
        previous_layer=b3, filters=encoder_filters[3], conv2d_layers=3, batchNorm=True)
    encoder = own_vgg16_encoder_block(
        previous_layer=b4, filters=encoder_filters[4], conv2d_layers=3, batchNorm=True)

    f1 = Flatten()(encoder)
    d1 = Dense(4096, activation='relu')(f1)
    # bottleneck = Dense(4096, activation='relu')(bottleneck)
    bottleneck = Dense(dense_size, activation='relu')(d1)
    # bottleneck = Dense(4096, activation='relu')(bottleneck)
    d2 = Dense(4032, activation='relu')(bottleneck)
    reshape = Reshape((12, 12, 28))(d2)

    b5 = own_vgg16_decoder_block(
        previous_layer=reshape, filters=decoder_filters[0], conv2d_layers=3, batchNorm=True)
    b6 = own_vgg16_decoder_block(
        previous_layer=b5, filters=decoder_filters[1], conv2d_layers=3, batchNorm=True)
    b7 = own_vgg16_decoder_block(
        previous_layer=b6, filters=decoder_filters[2], conv2d_layers=3, batchNorm=True)
    b8 = own_vgg16_decoder_block(
        previous_layer=b7, filters=decoder_filters[3], conv2d_layers=2, batchNorm=True)
    decoder = own_vgg16_decoder_block(
        previous_layer=b8, filters=decoder_filters[4], conv2d_layers=2, batchNorm=True)

    decoder = Conv2D(filters=1, kernel_size=(3, 3),
                     padding="same", activation="sigmoid")(decoder)

    return decoder, bottleneck
