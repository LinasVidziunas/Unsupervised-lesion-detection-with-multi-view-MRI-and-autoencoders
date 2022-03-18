from keras.layers import Conv2D, Reshape, Flatten, Dense
from keras.layers import UpSampling2D, MaxPool2D
from keras.layers import BatchNormalization, Dropout

from variational import Sampling


def own_vgg16_conv2d_block(previous_layer, filters, batchNorm: bool):
    x = Conv2D(filters=filters, kernel_size=(3, 3),
               padding="same", activation="relu")(previous_layer)
    if batchNorm:
        x = BatchNormalization()(x)

    return x


def own_vgg16_encoder_block(previous_layer, filters: int,
                          conv2d_layers: int = 2,
                          batchNorm: bool = False,
                          dropout_rate: float = 0,
                          max_pool: bool = True):

    block = own_vgg16_conv2d_block(previous_layer, filters, batchNorm)

    for _ in range(conv2d_layers-1):
        block = own_vgg16_conv2d_block(block, filters, batchNorm)

    if dropout_rate != 0:
        block = Dropout(dropout_rate)(block)

    if max_pool:
        block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block)

    return block


def own_vgg16_decoder_block(previous_layer, filters: int,
                          conv2d_layers: int = 2,
                          batchNorm: bool = False,
                          dropout_rate: float = 0,
                          up_sampling: bool = True):
    block = previous_layer

    if up_sampling:
        block = UpSampling2D(size=(2, 2))(block)

    if dropout_rate != 0:
        block = Dropout(dropout_rate)(block)

    for _ in range(conv2d_layers):
        block = own_vgg16_conv2d_block(block, filters, batchNorm)

    return block


def own_vgg16(inputs, dropout_rate: float = 0, batchNorm: bool = True, include_top: bool = True, dense_size: int = 120, latent_filters: int = 512):
    """
    :param: dropout_rate: dropout rate between all encoder and decoder blocks
    :param: batchNorm: BatchNorm after each conv layer
    :param: include_top: whether to include dense layer as bottleneck
    :param: dense_size: used if include_top is True, defines amount of units for dense layer bn
    :param: latent_filters: number of filters in the last encoder block

    Returns the full autoencoder and encoder part.
    """

    encoder_filters = [64, 128, 256, 512, 512]
    decoder_filters = [512, 512, 256, 128, 64]

    b1 = own_vgg16_encoder_block(
        previous_layer=inputs, filters=encoder_filters[0], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b2 = own_vgg16_encoder_block(
        previous_layer=b1, filters=encoder_filters[1], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b3 = own_vgg16_encoder_block(
        previous_layer=b2, filters=encoder_filters[2], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b4 = own_vgg16_encoder_block(
        previous_layer=b3, filters=encoder_filters[3], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)

    if include_top:
        b5 = own_vgg16_encoder_block(
            previous_layer=b4, filters=encoder_filters[4], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate, max_pool=True)
        encoder = own_vgg16_conv2d_block(previous_layer=b5, filters=latent_filters, batchNorm=batchNorm)

        f1 = Flatten()(encoder)

        bottleneck = Dense(dense_size, activation='relu')(f1)

        reshape = Reshape((12, 12, 28))(bottleneck)

        b5 = own_vgg16_decoder_block(
            previous_layer=reshape, filters=decoder_filters[0], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate, up_sampling=True)
    else:
        b5 = own_vgg16_encoder_block(
            previous_layer=b4, filters=encoder_filters[4], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate, max_pool=False)
        encoder = own_vgg16_conv2d_block(previous_layer=b5, filters=latent_filters, batchNorm=batchNorm)

        bottleneck = encoder

        b5 = own_vgg16_decoder_block(
            previous_layer=encoder, filters=decoder_filters[0], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate, up_sampling=False)

    b6 = own_vgg16_decoder_block(
        previous_layer=b5, filters=decoder_filters[1], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b7 = own_vgg16_decoder_block(
        previous_layer=b6, filters=decoder_filters[2], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b8 = own_vgg16_decoder_block(
        previous_layer=b7, filters=decoder_filters[3], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    decoder = own_vgg16_decoder_block(
        previous_layer=b8, filters=decoder_filters[4], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)

    decoder = Conv2D(filters=1, kernel_size=(3, 3),
                     padding="same", activation="sigmoid")(decoder)

    return decoder, bottleneck


def model_VAE_VGG16(inputs,
                  encoder_filters=[64, 128, 256, 512, 512],
                  decoder_filters=[512, 512, 256, 128, 64],
                  latent_conv_filters: int = 16,
                  latent_dim: int = 500,
                  batchNorm:bool = False,
                  dropout_rate:int = 0,):
    
    b1 = own_vgg16_encoder_block(
        previous_layer=inputs, filters=encoder_filters[0], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b2 = own_vgg16_encoder_block(
        previous_layer=b1, filters=encoder_filters[1], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b3 = own_vgg16_encoder_block(
        previous_layer=b2, filters=encoder_filters[2], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b4 = own_vgg16_encoder_block(
        previous_layer=b3, filters=encoder_filters[3], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    
    b5 = own_vgg16_encoder_block(
        previous_layer=b4, filters=encoder_filters[4], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate, max_pool=False)
    encoder = own_vgg16_conv2d_block(previous_layer=b5, filters=latent_conv_filters, batchNorm=batchNorm)
    
    flatten = Flatten()(encoder)
    
    dense_pre_bn = Dense(1000, name="dense_pre_bn")(flatten)
    
    z_mean = Dense(latent_dim, name="z_mean")(dense_pre_bn)
    z_log_var = Dense(latent_dim, name="z_log_var")(dense_pre_bn)
    z = Sampling(name="z")([z_mean, z_log_var])
        
    dense_post_bn = Dense(9216, name="dense_post_bn")(z)
    
    reshape = Reshape((24, 24, latent_conv_filters))(dense_post_bn)
    
    b5 = own_vgg16_decoder_block(
        previous_layer=reshape, filters=decoder_filters[0], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate, up_sampling=False)
    b6 = own_vgg16_decoder_block(
        previous_layer=b5, filters=decoder_filters[1], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b7 = own_vgg16_decoder_block(
        previous_layer=b6, filters=decoder_filters[2], conv2d_layers=3, batchNorm=batchNorm, dropout_rate=dropout_rate)
    b8 = own_vgg16_decoder_block(
        previous_layer=b7, filters=decoder_filters[3], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    decoder = own_vgg16_decoder_block(
        previous_layer=b8, filters=decoder_filters[4], conv2d_layers=2, batchNorm=batchNorm, dropout_rate=dropout_rate)
    
    output = Conv2D(filters=1, kernel_size=(3, 3),
                    padding="same", activation="sigmoid")(decoder)
    
    return output, z_mean, z_log_var, z
