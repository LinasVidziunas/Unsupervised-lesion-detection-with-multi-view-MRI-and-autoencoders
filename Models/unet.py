from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, concatenate


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
