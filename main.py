from numpy.random import seed
seed(420)
from tensorflow.random import set_seed
set_seed(420)

from tensorflow import constant
from tensorflow.keras.optimizers import Adam
from keras import Model
from keras.models import load_model
from keras.layers import Input
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanSquaredError

from results import ModelResults, get_roc, get_auc, default_save_data
from processed import get_data_by_patients
from variational import VAE, Sampling
from classification import Classification_using_transfer_learning
from Models.unet import model_MV_cAE_UNET
# from callbacks import ResultsCallback, AUCcallback

from numpy import array, concatenate

from os import path

# ------------------------ CONSTANTS ------------------------ #
# Configure these for each autoencoder!
# This will get used to save and load weights, and saving results.

IMAGE_DIM = [384, 384, 1]

# Change this to the desired name of your model.
# Used to identify the model!
MODEL_NAME = "multi_view_cae_test_unet"

# Epochs for the base autoencoder
EPOCHS = 25

# For all; autoencoder, classification via transfer learning,
# and also for fine tuning classification via transfer learning,
# but for fine tuning LEARNING_RATE * 1e-1
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

# Autoencoder base
ax_inputs, cor_inputs, sag_inputs = [Input((IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2])) for _ in range(3)]
inputs = [ax_inputs, cor_inputs, sag_inputs]

ax_output, cor_output, sag_output, encoder = model_MV_cAE_UNET(inputs)
autoencoder = Model([ax_inputs, cor_inputs, sag_inputs], [ax_output, cor_output, sag_output])

# Specific settings for transfer learning
CLASSIF_TF_BS = 32  # Batch size for classification via transfer learning
CLASSIF_TF_FT_BS = 32  # Batch size for fine tuning part of the classification via transfer learning

# ------------------------ Data path ----------------------- #
patients = get_data_by_patients(path_to_sets_folder="../sets/", image_dim=(IMAGE_DIM[0], IMAGE_DIM[1]))

x_train = list(patients["train"].values())
x_val = list(patients["validation"]["x"].values())
y_val = list(patients["validation"]["y"].values())
x_test = list(patients["test"]["x"].values())
y_test = list(patients["test"]["y"].values())
x_val_normal = list(patients["val_normal"].values())
x_val_abnormal = list(patients["val_abnormal"].values())
x_test_normal = list(patients["test_normal"].values())
x_test_abnormal = list(patients["test_abnormal"].values())

# ---------------------- BASE MODEL ---------------------- #
# Some constants used to name saved model
model_path = path.join('pre-trained_models', f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e.h5")

results = ModelResults(f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e")

if path.exists(model_path):
    print(f"\n\n-------------------------- LOADING PRE-TRAINED MODEL from {model_path} --------------------------\n\n")
    autoencoder = load_model(model_path, custom_objects={"VAE": VAE, "Sampling": Sampling}, compile=False)
else:
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                        loss=[BinaryCrossentropy(), BinaryCrossentropy(), BinaryCrossentropy()],
                        loss_weights=[1, 1, 1],
                        metrics=[MeanSquaredError()])

    print(autoencoder.summary())

    # Save autoencoder summary
    with open(
            path.join(
                results.save_in_dir,
                f"AE_summary{results.timestamp_string()}.txt"),
            'w') as f:
        autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

    #callbacks = [
        #ResultsCallback(f"{MODEL_NAME}_{BATCH_SIZE}bs_{EPOCHS}e",
        #                IMAGE_DIM, validation_dataset,
        #                save_at_epochs=[10, 25, 50, 100, 200, 300, 500, 1000, 1500, 2000]),
        #AUCcallback(results, validation_dataset, IMAGE_DIM, 10)]

    autoencoder_history = autoencoder.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val),
        #callbacks=callbacks,
    )

    print(f"\n\n---------------------------- SAVING PRE-TRAINED MODEL to {model_path} ----------------------------\n\n")
    autoencoder.save(model_path, save_format='h5')

    default_save_data(autoencoder_history, autoencoder, results, IMAGE_DIM,
                      x_val,
                      x_val_abnormal,
                      x_val_normal,
                      mse_keys=["axial_mean_squared_error", "coronal_mean_squared_error", "sagittal_mean_squared_error"],
                      val_mse_keys=["val_axial_mean_squared_error", "val_coronal_mean_squared_error", "val_sagittal_mean_squared_error"],
                      loss_keys=["axial_loss", "coronal_loss", "sagittal_loss"],
                      val_loss_keys=["val_axial_loss", "val_coronal_loss", "val_sagittal_loss"],
                      views=["axial", "coronal", "sagittal"])

# ------------------- Classification with IQR method ------------------- #
# iqr_method = IQR_method(autoencoder, x_val, y_val, x_test, y_test, IMAGE_DIM)
# threshold = iqr_method.obtain_threshold()
# predicted = iqr_method.classify(threshold)
#
# metr = Metrics([x[1] for x in y_test], predicted)
# threshold_results = metr.get_results()
# results.plot_confusion_matrix(metr.get_confusionmatrix())
# results.save_raw_data([f"Threshold: {threshold}"] + threshold_results, "iqr_method_results")
#
# # ------------------------- Model Evaluation --------------------------- #
# # Obtaining more specific data from the test set
# x_test_abnormal = test_dataset.get_abnormal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
# x_test_normal = test_dataset.get_normal_slices_as_normalized_pixel_arrays(shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
#
# test_abnormal_decoded = autoencoder.predict(x_test_abnormal)
# if isinstance(test_abnormal_decoded, tuple):
#     test_abnormal_decoded = test_abnormal_decoded[0]
#
# test_normal_decoded = autoencoder.predict(x_test_normal)
# if isinstance(test_normal_decoded, tuple):
#     test_normal_decoded = test_normal_decoded[0]
#
# test_normal_loss = mse(test_normal_decoded.reshape(len(test_normal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
#                        x_test_normal.reshape(len(x_test_normal), IMAGE_DIM[0] * IMAGE_DIM[1]))
# test_abnormal_loss = mse(test_abnormal_decoded.reshape(len(test_abnormal_decoded), IMAGE_DIM[0] * IMAGE_DIM[1]),
#                          x_test_abnormal.reshape(len(x_test_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))
#
# # Getting ROC
# fpr, tpr, thresholds = get_roc(test_abnormal_loss, test_normal_loss)
# auc_score = get_auc(fpr, tpr)
#
# # Getting results for every threshold
# results_thresholds = []
# for threshold in thresholds:
#     results_thresholds.append(Metrics([x[1] for x in y_test], iqr_method.classify(threshold)))
#
# # Saving the figures for each metric for each treshold
# results.plot_roc_curve(fpr, tpr, auc_score)
# results.plot_specificity(thresholds, results_thresholds)
# results.plot_sensitivity(thresholds, results_thresholds)
# results.plot_accuracy(thresholds, results_thresholds)
# results.plot_f1(thresholds, results_thresholds)
#
# ------------------- TRANSFER LEARNING ------------------- #
encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(
    autoencoder, encoder, inputs, x_val, y_val, x_test, y_test, views=3)

# Copy weights from autoencoder to encoder model
transfer_learning_classif.copy_weights()

classif_results = transfer_learning_classif.run(flatten_layer=True, learning_rate=LEARNING_RATE,
                                                batch_size=CLASSIF_TF_BS, epochs=20)

fine_tune_results = transfer_learning_classif.fine_tune(learning_rate=LEARNING_RATE * 1e-1, batch_size=CLASSIF_TF_FT_BS,
                                                        epochs=10, num_layers=0)

predictions = transfer_learning_classif.classif.predict(x_test)

for j, view in enumerate(['axial', 'coronal', 'sagittal']):
    test_abnormal_pred = []
    test_normal_pred = []

    for i, pred in enumerate(predictions[j]):
        if (y_test[j][i] == [0, 1]).all():
            test_abnormal_pred.append(pred)
        elif (y_test[j][i] == [1, 0]).all():
            test_normal_pred.append(pred)

    # Save classification via transfer learning predictions
    results.save_raw_data(predictions[j], f"classification_transfer_learning_predictions_{view}")
    results.save_raw_data(y_test[j], f"classification_transfer_learning_labels_{view}")

    for i, prediction in enumerate(predictions[j], start=0):
        print(f"Predicted: {prediction}. Correct: {y_test[j][i]}")

    # Getting ROC
    fpr, tpr, thresholds = get_roc([el[1] for el in test_abnormal_pred], [el[1] for el in test_normal_pred])
    auc_score = get_auc(fpr, tpr)

    results.plot_roc_curve(fpr, tpr, auc_score, f"classification_transfer_learning_ROC_curve_{view}")
    results.scatter_plot_of_predictions(predictions[j], y_test[j])
