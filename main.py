from processed import ProcessedData
from results import ModelResults
from Models.our import ourBestModel
from Models.unet import unet_org, unet_dense, unet_safe, unet_org_dense
from Models.vgg16_ae import vgg16, vgg16_dense, own_vgg16

import tensorflow 
from keras import losses, Model
from keras.layers import Input, Dense, Dropout
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy

from os import path

def default_save_data(history, autoencoder, model_results):
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


# if __name__ == "__main__":

# Change this to the desired name of your model.
# Used to identify the model in results.
MODEL_NAME = "original_unet_denseembedding"

# Define the dominant image dimensions
IMAGE_DIM = [384, 384, 1]

# Retrieve processed data
data = ProcessedData("../sets/")

x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of training images: {len(x_train)}") # Debugging

x_val = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of validation images: {len(x_val)}") # Debugging
y_val = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.validation.axial.slices]
y_val = tensorflow.constant(y_val, shape=(len(y_val), 2))

x_test = data.test.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of test images: {len(x_test)}") # Debugging
y_test = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.test.axial.slices]
y_test = tensorflow.constant(y_test, shape=(len(y_test), 2))

# Build the model
inputs = Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2]))

# autoencoder = ourBestModel()
# autoencoder = unet_dense(input_size=(384, 384, 1), skip_connections=False)
#autoencoder = unet_org(input_size=(384, 384, 1))
autoencoder = unet_org_dense(input_size=(384, 384, 1))
# autoencoder = vgg16(input_size=(384, 384, 1))
# autoencoder = vgg16_dense(input_size=(384, 384, 1), dense_size=120)
# autoencoder = unet_safe(None, input_size=(384, 384, 1))
# autoencoder = own_vgg16(input_shape=(384, 384, 1))

autoencoder = Model(inputs, outputs)

autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                    loss="binary_crossentropy",
                    metrics=[MeanSquaredError()])

autoencoder_results = ModelResults(MODEL_NAME)
    
with open(
        path.join(
            autoencoder_results.save_in_dir,
            f"AE_summary{autoencoder_results.timestamp_string()}.txt"),
        'w') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

autoencoder_history = autoencoder.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, x_val),
    )

default_save_data(autoencoder_history, autoencoder, autoencoder_results)

classif_results = ModelResults(MODEL_NAME)

encoder = Model(inputs, encoder)

# Copy over weigts
[encoder.layers[i].set_weights(autoencoder.layers[i].get_weights()) for i in range(0, len(encoder.layers)-1)]

# Freeze encoder
encoder.trainabe = False

# New model on top
x = encoder(inputs, training=False)
x = Dropout(0.2)(x)
x = Dense(2, activation='softmax', name="classification")(x)
classif = Model(inputs, x)

classif.compile(
    optimizer=tensorflow.keras.optimizers.Adam(),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy()])

classif.summary()

with open(
        path.join(
            autoencoder_results.save_in_dir,
            f"Classif_summary{classif_results.timestamp_string()}.txt"),
        'w') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

classif_history = classif.fit(
    x_val,
    y_val,
    epochs=20,
    validation_data=(x_test, y_test))

# Fine tunings
encoder.trainable = True

classif.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(),
    metrics=[CategoricalAccuracy()])

fine_classif_history = classif.fit(
    x_val,
    y_val,
    epochs=10,
    validation_data=(x_test, y_test))

