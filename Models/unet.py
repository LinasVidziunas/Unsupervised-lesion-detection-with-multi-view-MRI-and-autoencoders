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


# Folger ikke standard unet arkitetktur!

# def unet_dense(inputs, dense_size: int = 120):
#     c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#     c1skip = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1skip)

#     c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     c2skip = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2skip)

#     c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     c3skip = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3skip)

#     c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     c4skip = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     p4 = MaxPooling2D(pool_size=(2, 2))(c4skip)

#     c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#     c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#     flatten = Flatten()(c5)
#     d1 = Dense(576, activation="relu")(flatten)
#     bottle = Dense(dense_size, activation='relu')(d1)
#     d2 = Dense(576, activation='relu')(bottle)
#     reshape = Reshape((24, 24, 1))(d2)

#     u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(reshape)
#     u6 = concatenate([u6, c4skip])
#     c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

#     u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3skip])
#     c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

#     u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2skip])
#     c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

#     u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1skip])
#     c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

#     output = Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

#     return output, bottle


# def unet_dropout(inputs, dropout_rate: float = 0.35):
#     c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
#     c1skip = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
#     c1skip = Dropout(dropout_rate)(c1skip)
#     p1 = MaxPooling2D((2, 2))(c1skip)

#     c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
#     c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     c2skip = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
#     c2skip = Dropout(dropout_rate)(c2skip)
#     p2 = MaxPooling2D((2, 2))(c2skip)

#     c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
#     c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     c3skip = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
#     c3skip = Dropout(dropout_rate)(c3skip)
#     p3 = MaxPooling2D((2, 2))(c3skip)

#     c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
#     c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     c4skip = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
#     c4skip = Dropout(dropout_rate)(c4skip)
#     p4 = MaxPooling2D(pool_size=(2, 2))(c4skip)

#     drop4 = Dropout(dropout_rate)(p4)
#     c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
#     bottle = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(drop4)
#     c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bottle)
#     drop5 = Dropout(dropout_rate)(c5)

#     u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
#     u6 = concatenate([u6, c4skip])
#     c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
#     c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

#     u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3skip])
#     c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
#     c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

#     u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2skip])
#     c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
#     c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

#     u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1skip])
#     c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
#     c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

#     output = Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

#     return output, bottle
