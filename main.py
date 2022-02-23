import tensorflow
from keras import losses, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy

from processed import ProcessedData
from results import ModelResults
from Models.our import ourBestModel
from Models.unet import unet_org, unet_org_dense, new_unet_org_dense
from Models.vgg16_ae import vgg16, vgg16_dense, own_vgg16
from classification import Classification_using_transfer_learning

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
MODEL_NAME = "classified_unet_1000_epochs"

# Define the dominant image dimensions
IMAGE_DIM = [384, 384, 1]

# Retrieve processed data
data = ProcessedData("../sets/")

x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of training images: {len(x_train)}")

x_val = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of validation images: {len(x_val)}")
y_val = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.validation.axial.slices]
y_val = tensorflow.constant(y_val, shape=(len(y_val), 2))

x_test = data.test.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of test images: {len(x_test)}")
y_test = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.test.axial.slices]
y_test = tensorflow.constant(y_test, shape=(len(y_test), 2))

# Build the model
inputs = Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2]))

outputs, encoder = unet_org(inputs)

autoencoder = Model(inputs, outputs)

autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                    loss="binary_crossentropy",
                    metrics=[MeanSquaredError()])

results = ModelResults(MODEL_NAME)
print(autoencoder.summary())

# Save autoencoder summary
with open(
        path.join(
            results.save_in_dir,
            f"AE_summary{results.timestamp_string()}.txt"),
        'w') as f:
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

autoencoder_history = autoencoder.fit(
    x_train,
    x_train,
    epochs=1000,
    batch_size=32,
    validation_data=(x_val, x_val),
    )

default_save_data(autoencoder_history, autoencoder, results)


encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(autoencoder, encoder, inputs, x_val, y_val, x_test, y_test)

# Copy weights from autoencoder to encoder model
transfer_learning_classif.copy_weights()

classif_results = transfer_learning_classif.run(flatten_layer=True, learning_rate=1e-5, batch_size=32, epochs=40)

fine_tune_results = transfer_learning_classif.fine_tune(learning_rate=1e-5, batch_size=32, epochs=10, num_layers=5)

predictions = transfer_learning_classif.classif.predict(x_test)

# Save classification via transfer learning predictions
raw = "prediction,correct"
for i, prediction in enumerate(predictions, start=0):
    raw += f"\n{prediction},{y_test[i]}"
    print(f"Predicted: {prediction}. Correct: {y_test[i]}")

with open(
        path.join(
            results.save_in_dir,
            f"classificatoin_predicted_vs_correct{results.timestamp_string()}.raw"),
        'w') as f:
    f.write(raw)

results.scatter_plot_of_predictions(predictions, y_test)
