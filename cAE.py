from processed import ProcessedData
from plotting import ModelPlotting

from keras import layers, Model, losses
from keras.layers import Flatten, Dense, Reshape, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D, LeakyReLU
from keras.metrics import MeanSquaredError


data = ProcessedData("../sets/")
x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(384, 384))
x_test = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(384, 384))

input = layers.Input(shape=(384, 384, 1))

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

x = Dense(576)(x)
x = LeakyReLU(alpha=0.2)(x)

x = Reshape((24, 24, 1))(x)

x = UpSampling2D((2, 2))(x)
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
                    metrics=[MeanSquaredError()])
autoencoder.summary()

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=1000,
    batch_size=32,
    validation_data=(x_test, x_test),
)

test_abnormal = data.validation.axial.get_abnormal_slices_as_normalized_pixel_arrays(
    shape=(384, 384))
test_normal = data.validation.axial.get_normal_slices_as_normalized_pixel_arrays(
    shape=(384, 384))

# Plotting the MSE distrubution of normal slices
decoded_normal = autoencoder.predict(test_normal)

loss_normal = losses.mse(decoded_normal.reshape(len(test_normal), 384 * 384),
                         test_normal.reshape(len(test_normal), 384 * 384))

decoded_abnormal = autoencoder.predict(test_abnormal)
loss_abnormal = losses.mse(
    decoded_abnormal.reshape(len(test_abnormal), 384 * 384),
    test_abnormal.reshape(len(test_abnormal), 384 * 384))

plot = ModelPlotting(history, save_in_dir="sets")

plot.plot_mse_train_vs_val()
plot.plot_loss_train_vs_val()

plot.histogram_mse_loss(loss_normal, loss_abnormal)
plot.histogram_mse_loss_seperate(loss_normal, loss_abnormal)

reconstructed_images = autoencoder.predict(x_test)

plot.input_vs_reconstructed_images(
    [el.reshape(384, 384) for el in x_test],
    [el.reshape(384, 384) for el in reconstructed_images]
)
