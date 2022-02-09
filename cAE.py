from processed import ProcessedData
from plotting import ModelPlotting
from Models.our import ourBestModel
from Models.unet import unet, unet_dense, unet_safe
from Models.vgg16_ae import vgg16, vgg16_dense, own_vgg16

import tensorflow
from keras import losses
from keras.losses import MeanSquaredError


data = ProcessedData("../sets/")
x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(384, 384))
x_test = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(384, 384))

# autoencoder = ourBestModel()
# autoencoder = unet_dense(input_size=(384, 384, 1), skip_connections=False)
autoencoder = unet_dense()
# autoencoder = vgg16(input_size=(384, 384, 1))
# autoencoder = vgg16_dense(input_size=(384, 384, 1), dense_size=120)
# autoencoder = unet_safe(None, input_size=(384, 384, 1))
#autoencoder = own_vgg16(input_shape=(384, 384, 1))

autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
                    loss="binary_crossentropy",
                    metrics=[MeanSquaredError()])
autoencoder.summary()

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=200,
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
