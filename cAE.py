from processed import ProcessedData
from results import ModelResults
from Models.our import ourBestModel
from Models.unet import unet, unet_dense, unet_safe
from Models.vgg16_ae import vgg16, vgg16_dense, own_vgg16

import tensorflow
from keras import losses
from keras.losses import MeanSquaredError

from os import path

# Change this to the desired name of your model.
# Used to identify the model in results.
MODEL_NAME = "test"

IMAGE_DIM = [384, 384]

data = ProcessedData("../sets/")
x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
x_val = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

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

model_results = ModelResults(MODEL_NAME)

with open(
        path.join(
            model_results.save_in_dir,
            f"AE_summary{model_results.timestamp_string()}.txt"),
        'w') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_val, x_val),
)
model_results.save_raw_data(history.history['mean_squared_error'], "mse_per_epoch")
model_results.save_raw_data(history.history['val_mean_squared_error'], "val_mse_per_epoch")
model_results.save_raw_data(history.history['loss'], "loss_epoch")
model_results.save_raw_data(history.history['val_loss'], "val_loss_epoch")

validation_abnormal = data.validation.axial.get_abnormal_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
validation_normal = data.validation.axial.get_normal_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

# Plotting the MSE distrubution of normal slices
decoded_normal = autoencoder.predict(validation_normal)
loss_normal = losses.mse(decoded_normal.reshape(len(validation_normal), IMAGE_DIM[0] * IMAGE_DIM[1]),
                         validation_normal.reshape(len(validation_normal), IMAGE_DIM[0] * IMAGE_DIM[1]))

# Saving raw MSE loss of normal slices
model_results.save_raw_data(loss_normal, "normal_mse_loss")

decoded_abnormal = autoencoder.predict(validation_abnormal)
loss_abnormal = losses.mse(
    decoded_abnormal.reshape(len(validation_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]),
    validation_abnormal.reshape(len(validation_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))

# Saving raw MSE loss of abnormal slices
model_results.save_raw_data(loss_abnormal, "abnormal_mse_loss")

model_results.plot_mse_train_vs_val(history)
model_results.plot_loss_train_vs_val(history)

model_results.histogram_mse_loss(loss_normal, loss_abnormal)
model_results.histogram_mse_loss_seperate(loss_normal, loss_abnormal)

reconstructed_images = autoencoder.predict(x_val)

model_results.input_vs_reconstructed_images(
    [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in x_val],
    [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in reconstructed_images]
)
