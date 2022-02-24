import tensorflow
from keras import Model
from keras.layers import Input
from keras.losses import MeanSquaredError, BinaryCrossentropy

from processed import ProcessedData
from results import ModelResults, default_save_data
from Models.unet import unet_org
from classification import Classification_using_transfer_learning

from os import path


# ------------------------ CONSTANTS ------------------------ #
# Change this to the desired name of your model.
# Used to identify the model in results.
MODEL_NAME = "classified_unet_1000_epochs"

# Define the dominant image dimensions
IMAGE_DIM = [384, 384, 1]


# --------------------- IMPORTING DATA --------------------- #
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


# ---------------------- BASE MODEL ---------------------- #
inputs = Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2]))

outputs, encoder = unet_org(inputs)

autoencoder = Model(inputs, outputs)

autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=BinaryCrossentropy(),
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

default_save_data(autoencoder_history, autoencoder, results, IMAGE_DIM, data.validation.axial)


# ------------------- TRANSFER LEARNING ------------------- #
encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(autoencoder, encoder, inputs, x_val, y_val, x_test, y_test)

# Copy weights from autoencoder to encoder model
transfer_learning_classif.copy_weights()

classif_results = transfer_learning_classif.run(flatten_layer=True, learning_rate=1e-5, batch_size=32, epochs=40)

fine_tune_results = transfer_learning_classif.fine_tune(learning_rate=1e-5, batch_size=32, epochs=10, num_layers=5)

predictions = transfer_learning_classif.classif.predict(x_test)

# Save classification via transfer learning predictions
results.save_raw_data(predictions, "classification_tl_predictions")
results.save_raw_data(y_test, "classififcation_tl_labels")

for i, prediction in enumerate(predictions, start=0):
    print(f"Predicted: {prediction}. Correct: {y_test[i]}")

results.scatter_plot_of_predictions(predictions, y_test)
