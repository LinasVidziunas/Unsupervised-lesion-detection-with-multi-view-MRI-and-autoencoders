import numpy as np
import tensorflow
from tensorflow.keras.optimizers import Adam
from keras import Model
from keras.models import load_model
from keras.layers import Input
from keras.losses import MeanSquaredError, BinaryCrossentropy, mse
from results import Metrics, get_roc, plot_specificity, plot_sensitivity, plot_f1, plot_accuracy

from processed import ProcessedData
from results import ModelResults, default_save_data
from Models.vgg16_ae import own_vgg16
from classification import Classification_using_transfer_learning
from callbacks import ResultsCallback

from sklearn import metrics as skmetrics

from os import path

# ------------------------ CONSTANTS ------------------------ #
# Configure these for each autoencoder!
# This will get used to save and load weights, and saving results.

# Epochs for the base autoencoder
EPOCHS = 2500

# Change this to the desired name of your model.
# Used to identify the model!
MODEL_NAME = "VGG16_axial_30_dr_50_drbn_batchNorm_300bn_es_00069explrs"

# Define the dominant image dimensions
IMAGE_DIM = [384, 384, 1]

# Autoencoder base
inputs = Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2]))
outputs, encoder = own_vgg16(inputs, dropout_rate=0.3, dropout_rate_bn=0.5, batchNorm=True, dense_size=300)

# --------------------- IMPORTING DATA --------------------- #
data = ProcessedData("../sets/")

x_train = data.train.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of training images: {len(x_train)}")

x_val = data.validation.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of validation images: {len(x_val)}")

y_val = [[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in
         data.validation.axial.slices]
y_val = tensorflow.constant(y_val, shape=(len(y_val), 2))

x_test = data.test.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of test images: {len(x_test)}")

y_test = [[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.test.axial.slices]
y_test = tensorflow.constant(y_test, shape=(len(y_test), 2))


# ---------------------- BASE MODEL ---------------------- #
# Some constants used to name saved model
batch_size = 32
model_path = path.join('pre-trained_models', f"{MODEL_NAME}_{batch_size}bs_{EPOCHS}e.h5")

autoencoder = Model(inputs, outputs, name=MODEL_NAME)
results = ModelResults(f"{MODEL_NAME}_{batch_size}bs_{EPOCHS}e")

if path.exists(model_path):
    print(f"\n\n-------------------------- LOADING PRE-TRAINED MODEL from {model_path} --------------------------\n\n")
    autoencoder = load_model(model_path, compile=False)
else:
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3),
                        loss=BinaryCrossentropy(),
                        metrics=[MeanSquaredError()])

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
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=(x_val, x_val),
        callbacks=[ResultsCallback(f"{MODEL_NAME}_{batch_size}bs_{EPOCHS}e", IMAGE_DIM, data.validation.axial, save_at_epochs=[10, 25, 50, 100, 200, 300])],
    )

    print(f"\n\n---------------------------- SAVING PRE-TRAINED MODEL to {model_path} ----------------------------\n\n")
    autoencoder.save(model_path, save_format='h5')

    default_save_data(autoencoder_history, autoencoder, results, IMAGE_DIM, data.validation.axial)

# ------------------- Threshold selection and evaluation ------------------- #
# Obtaining validation set losses and labels
validation_set_labels = [_slice.get_abnormality() for _slice in data.validation.axial.slices]
validation_decoded = autoencoder.predict(x_val)
validation_loss = mse(validation_decoded.reshape(len(validation_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
                      x_val.reshape(len(x_val), IMAGE_DIM[0] * IMAGE_DIM[1]))
# Obtain threshold
q3, q1 = np.percentile(validation_loss, [75, 25])
iqr = q3 - q1
threshold = q3 + 1.5*iqr

#Get test data
test_set_labels = [_slice.get_abnormality() for _slice in data.test.axial.slices]
test_decoded = autoencoder.predict(x_test)
test_loss = mse(
    test_decoded.reshape(len(test_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
    x_test.reshape(len(x_test), IMAGE_DIM[0] * IMAGE_DIM[1]))

#Classify based on set threshold
predicted = []
for loss in test_loss:
    if loss < threshold:
        predicted.append(0)
    if loss > threshold:
        predicted.append(1)
#Getting results from that threshold
threshold_results = Metrics(test_set_labels, predicted).get_results()
#Obtaining more specific data from the test set
test_abnormal = data.validation.axial.get_abnormal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
test_normal = data.validation.axial.get_normal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

test_abnormal_decoded = autoencoder.predict(test_abnormal)
test_normal_decoded = autoencoder.predict(test_normal)

test_normal_loss = mse(test_normal_decoded.reshape(len(test_normal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
                       test_normal.reshape(len(test_normal), IMAGE_DIM[0] * IMAGE_DIM[1]))
test_abnormal_loss = mse(test_abnormal_decoded.reshape(len(test_abnormal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
                         test_abnormal.reshape(len(test_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))
#Getting ROC
fpr, tpr, thresholds = get_roc(test_abnormal_loss, test_normal_loss)
AUC_score = skmetrics.auc(fpr, tpr)
#Getting results for every threshold

results_thresholds = []
for threshold in thresholds:
    predicted_test = []
    for loss in test_loss:
        if loss < threshold:
            predicted.append(0)
        if loss > threshold:
            predicted.append(1)
    results_thresholds.append(Metrics(test_set_labels, predicted_test))
#Saving the figures for each metric for each treshold

plot_specificity(thresholds, results_thresholds)
plot_sensitivity(thresholds, results_thresholds)
plot_accuracy(thresholds, results_thresholds)
plot_f1(thresholds, results_thresholds)


# ------------------- TRANSFER LEARNING ------------------- #
encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(autoencoder, encoder, inputs, x_val, y_val, x_test,
                                                                   y_test)

# Copy weights from autoencoder to encoder model
transfer_learning_classif.copy_weights()

classif_results = transfer_learning_classif.run(flatten_layer=True, learning_rate=1e-4, batch_size=32, epochs=20)

fine_tune_results = transfer_learning_classif.fine_tune(learning_rate=1e-5, batch_size=32, epochs=10, num_layers=5)

predictions = transfer_learning_classif.classif.predict(x_test)

# Save classification via transfer learning predictions
results.save_raw_data(predictions, "classification_tl_predictions")
results.save_raw_data(y_test, "classififcation_tl_labels")

for i, prediction in enumerate(predictions, start=0):
    print(f"Predicted: {prediction}. Correct: {y_test[i]}")

results.scatter_plot_of_predictions(predictions, y_test)
