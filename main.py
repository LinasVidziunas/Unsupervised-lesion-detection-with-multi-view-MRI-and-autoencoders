from numpy.random import seed
seed(420)
from tensorflow.random import set_seed
set_seed(420)

import tensorflow
from tensorflow.keras.optimizers import Adam
from keras import Model
from keras.models import load_model
from keras.layers import Input
from keras.losses import BinaryCrossentropy, mse
from keras.metrics import MeanSquaredError

from results import ModelResults, default_save_data
from results import Metrics, get_roc, get_auc
from processed import ProcessedData
from callbacks import ResultsCallback, AUCcallback
from variational import VAE, Sampling
from classification import Classification_using_transfer_learning, IQR_method
from Models.vgg16_ae import model_VAE_VGG16
from bootstrapping import bootstrapping_mse, bootstrapping_TL

from os import path

# ------------------------ CONSTANTS ------------------------ #
# Configure these for each autoencoder!
# This will get used to save and load weights, and saving results.

# Define the dominant image dimensions
# [384, 384, 1] for axial
# [320, 320, 1] for sagittal and coronal
IMAGE_DIM = [384, 384, 1]

# Change this to the desired name of your model.
# Used to identify the model!
MODEL_NAME = "VAE_test"

# Epochs for the base autoencoder
EPOCHS = 30

# For all; autoencoder, classification via transfer learning,
# and also for fine tuning classification via transfer learning,
# but for fine tuning LEARNING_RATE * 1e-1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# Autoencoder base
inputs = Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2]))
outputs, z_mean, z_log_var, encoder = model_VAE_VGG16(inputs)
autoencoder = VAE(inputs, (outputs, z_mean, z_log_var, encoder), name=MODEL_NAME)

# Specific settings for transfer learning
CLASSIF_TF_BS = 32 # Batch size for classification via transfer learning
CLASSIF_TF_FT_BS = 32 # Batch size for fine tuning part of the classification via transfer learning


# ------------------------ Data path ----------------------- #
data = ProcessedData("../sets/")


# ----------------------- Define view -----------------------#
# Change these to change view!
train_dataset = data.train.axial
validation_dataset = data.validation.axial
test_dataset = data.test.axial


# ---------------- Loading data into memory -----------------#
x_train = train_dataset.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of training images: {len(x_train)}")

x_val = validation_dataset.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of validation images: {len(x_val)}")

y_val = [[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in
         validation_dataset.slices]
y_val = tensorflow.constant(y_val, shape=(len(y_val), 2))

x_test = test_dataset.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of test images: {len(x_test)}")

y_test = [[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in test_dataset.slices]
y_test = tensorflow.constant(y_test, shape=(len(y_test), 2))


# ---------------------- BASE MODEL ---------------------- #
# Some constants used to name saved model
model_path = path.join('pre-trained_models', f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e.h5")

results = ModelResults(f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e")

if path.exists(model_path):
    print(f"\n\n-------------------------- LOADING PRE-TRAINED MODEL from {model_path} --------------------------\n\n")
    autoencoder = load_model(model_path, custom_objects={"VAE": VAE, "Sampling": Sampling}, compile=False)
else:
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
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

    callbacks = [
        ResultsCallback(f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e",
                        IMAGE_DIM, validation_dataset,
                        save_at_epochs=[10, 25, 50, 100, 200, 300, 500, 1000, 1500, 2000]),
        AUCcallback(results, validation_dataset, IMAGE_DIM, 10)]

    autoencoder_history = autoencoder.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
    )

    print(f"\n\n---------------------------- SAVING PRE-TRAINED MODEL to {model_path} ----------------------------\n\n")
    autoencoder.save(model_path, save_format='h5')

    default_save_data(autoencoder_history, autoencoder, results, IMAGE_DIM, validation_dataset)


# ------------------- Classification with IQR method ------------------- #
iqr_method = IQR_method(autoencoder, x_val, y_val, x_test, y_test, IMAGE_DIM)
threshold = iqr_method.obtain_threshold()
predicted = iqr_method.classify(threshold)

metr = Metrics([x[1] for x in y_test], predicted)
threshold_results = metr.get_results()
results.plot_confusion_matrix(metr.get_confusionmatrix())
results.save_raw_data([f"Threshold: {threshold}"] + threshold_results, "iqr_method_results")


# ------------------------- Model Evaluation --------------------------- #
# Obtaining more specific data from the test set
x_test_abnormal = test_dataset.get_abnormal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
x_test_normal = test_dataset.get_normal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

test_abnormal_decoded = autoencoder.predict(x_test_abnormal)
if isinstance(test_abnormal_decoded, tuple):
    test_abnormal_decoded = test_abnormal_decoded[0]

test_normal_decoded = autoencoder.predict(x_test_normal)
if isinstance(test_normal_decoded, tuple):
    test_normal_decoded = test_normal_decoded[0]

test_normal_loss = mse(test_normal_decoded.reshape(len(test_normal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
                       x_test_normal.reshape(len(x_test_normal), IMAGE_DIM[0] * IMAGE_DIM[1]))
test_abnormal_loss = mse(test_abnormal_decoded.reshape(len(test_abnormal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
                         x_test_abnormal.reshape(len(x_test_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))

# Getting ROC
fpr, tpr, thresholds = get_roc(test_abnormal_loss, test_normal_loss)
auc_score = get_auc(fpr, tpr)

# Getting results for every threshold
results_thresholds = []
for threshold in thresholds:
    results_thresholds.append(Metrics([x[1] for x in y_test], iqr_method.classify(threshold)))

# Saving the figures for each metric for each treshold
results.plot_roc_curve(fpr, tpr, auc_score)
results.plot_specificity(thresholds, results_thresholds)
results.plot_sensitivity(thresholds, results_thresholds)
results.plot_accuracy(thresholds, results_thresholds)
results.plot_f1(thresholds, results_thresholds)

#Getting bootstrapping results with MSE losses
test_data = test_dataset.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
test_labels = [_slice.get_abnormality() for _slice in test_dataset.slices]

mean_auc, std_auc = bootstrapping_mse(autoencoder, test_data, test_labels, 100, IMAGE_DIM[0])
print("Mean auc MSE", mean_auc)
print("Std auc MSE", std_auc)

# ------------------- TRANSFER LEARNING ------------------- #
encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(
    autoencoder, encoder, inputs, x_val, y_val, x_test, y_test)

# Copy weights from autoencoder to encoder model
transfer_learning_classif.copy_weights()

classif_results = transfer_learning_classif.run(flatten_layer=True, learning_rate=LEARNING_RATE, batch_size=CLASSIF_TF_BS, epochs=20)

fine_tune_results = transfer_learning_classif.fine_tune(learning_rate=LEARNING_RATE*1e-1, batch_size=CLASSIF_TF_FT_BS, epochs=10, num_layers=5)

predictions = transfer_learning_classif.classif.predict(x_test)
test_normal_pred = transfer_learning_classif.classif.predict(x_test_normal)
test_abnormal_pred = transfer_learning_classif.classif.predict(x_test_abnormal)

# Save classification via transfer learning predictions
results.save_raw_data(predictions, "classification_transfer_learning_predictions")
results.save_raw_data(y_test, "classification_transfer_learning_labels")

for i, prediction in enumerate(predictions, start=0):
    print(f"Predicted: {prediction}. Correct: {y_test[i]}")

# Getting ROC
fpr, tpr, thresholds = get_roc([el[1] for el in test_abnormal_pred], [el[1] for el in test_normal_pred])
auc_score = get_auc(fpr, tpr)

# Get results with bootstrapping
mean_auc_TL, std_auc_TL = bootstrapping_TL(transfer_learning_classif, test_data, test_labels, 100)
print("Mean auc TL", mean_auc_TL)
print("Std auc TL", std_auc_TL)



results.plot_roc_curve(fpr, tpr, auc_score, "classification_transfer_learning_ROC_curve")
results.scatter_plot_of_predictions(predictions, y_test)
