from processed import ProcessedData
from plotting import ModelPlotting
from Models.our import ourBestModel
from Models.unet import unet, unet_dense, unet_safe
from Models.vgg16_ae import vgg16, vgg16_dense, own_vgg16

import tensorflow
from keras import losses
from keras.losses import MeanSquaredError

import numpy as np

data = ProcessedData("../sets/")
x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(shape=(384, 384))
x_test = data.validation.axial.get_slices_as_normalized_pixel_arrays(shape=(384, 384))
train_slices = data.train.axial.slices
test_slices = data.validation.axial.slices

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


test_abnormal_l = []
test_normal_l = []

for _slice in test_slices:
    if _slice.get_abnormality() == 1:
        test_abnormal_l.append(_slice)
    elif _slice.get_abnormality() == 0:
        test_normal_l.append(_slice)

test_abnormal = np.zeros((len(test_abnormal_l), 384, 384))
test_normal = np.zeros((len(test_normal_l), 384, 384))

for i, _slice in enumerate(test_abnormal_l):
    test_abnormal[i][:][:] = _slice.normalized_pixel_array()

for i, _slice in enumerate(test_normal_l):
    test_normal[i][:][:] = _slice.normalized_pixel_array()

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
