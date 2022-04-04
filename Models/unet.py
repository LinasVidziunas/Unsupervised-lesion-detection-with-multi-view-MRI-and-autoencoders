from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, concatenate


def unet_encoder(inputs, encoder_filters=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]):
    c1 = Conv2D(encoder_filters[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1skip = Conv2D(encoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1skip)
    c2 = Conv2D(encoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2skip = Conv2D(encoder_filters[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2skip)
    c3 = Conv2D(encoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3skip = Conv2D(encoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3skip)
    c4 = Conv2D(encoder_filters[6], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4skip = Conv2D(encoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4skip)
    c5 = Conv2D(encoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    bottleneck = Conv2D(encoder_filters[9], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    return bottleneck, c1skip, c2skip, c3skip, c4skip

def unet_decoder(bottleneck, c1skip, c2skip, c3skip, c4skip,
               decoder_filters=[512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64], name_output_layer="axial"):
    u6 = Conv2DTranspose(decoder_filters[0], (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u6 = concatenate([u6, c4skip])
    c6 = Conv2D(decoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(decoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(decoder_filters[3], (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3skip])
    c7 = Conv2D(decoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(decoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(decoder_filters[6], (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2skip])
    c8 = Conv2D(decoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(decoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(decoder_filters[9], (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1skip])
    c9 = Conv2D(decoder_filters[10], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(decoder_filters[11], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    output = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same', name=name_output_layer)(c9)

    return output

def model_MV_cAE_UNET(inputs: list,
                       encoder_filters=[[64, 64, 128, 128, 256, 256, 512, 512, 1024, 128],
                                        [64, 64, 128, 128, 256, 256, 512, 512, 1024, 32],
                                        [64, 64, 128, 128, 256, 256, 512, 512, 1024, 16]],
                       decoder_filters=[512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]):

    ax_bn, ax_c1skip, ax_c2skip, ax_c3skip, ax_c4skip = unet_encoder(inputs[0], encoder_filters=encoder_filters[0])
    cor_bn, cor_c1skip, cor_c2skip, cor_c3skip, cor_c4skip = unet_encoder(inputs[1], encoder_filters=encoder_filters[1])
    sag_bn, sag_c1skip, sag_c2skip, sag_c3skip, sag_c4skip = unet_encoder(inputs[2], encoder_filters=encoder_filters[2])

    bottleneck = concatenate([ax_bn, cor_bn, sag_bn], axis=3)

    ax_output = unet_decoder(bottleneck, ax_c1skip, ax_c2skip, ax_c3skip, ax_c4skip, decoder_filters, name_output_layer="axial")
    cor_output = unet_decoder(bottleneck, cor_c1skip, cor_c2skip, cor_c3skip, cor_c4skip, decoder_filters, name_output_layer="coronal")
    sag_output = unet_decoder(bottleneck, sag_c1skip, sag_c2skip, sag_c3skip, sag_c4skip, decoder_filters, name_output_layer="sagittal")

    return ax_output, cor_output, sag_output, bottleneck

def unet(inputs,
       encoder_filters=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024],
       decoder_filters=[512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]):

    c1 = Conv2D(encoder_filters[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1skip = Conv2D(encoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1skip)
    c2 = Conv2D(encoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2skip = Conv2D(encoder_filters[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2skip)
    c3 = Conv2D(encoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3skip = Conv2D(encoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3skip)
    c4 = Conv2D(encoder_filters[6], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4skip = Conv2D(encoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4skip)
    c5 = Conv2D(encoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    bottle = Conv2D(encoder_filters[9], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(decoder_filters[0], (2, 2), strides=(2, 2), padding='same')(bottle)
    u6 = concatenate([u6, c4skip])
    c6 = Conv2D(decoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(decoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(decoder_filters[3], (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3skip])
    c7 = Conv2D(decoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(decoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(decoder_filters[6], (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2skip])
    c8 = Conv2D(decoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(decoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(decoder_filters[9], (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1skip])
    c9 = Conv2D(decoder_filters[10], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(decoder_filters[11], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    output = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same')(c9)

    return output, bottle
