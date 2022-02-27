import tensorflow
from tensorflow.keras.optimizers import Adam

from keras import Model
from keras.models import load_model
from keras.layers import Input
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.callbacks import EarlyStopping, LearningRateScheduler

from processed import ProcessedData
from results import ModelResults, default_save_data
from Models.vgg16_ae import own_vgg16
from classification import Classification_using_transfer_learning

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

y_val = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.validation.axial.slices]
y_val = tensorflow.constant(y_val, shape=(len(y_val), 2))

x_test = data.test.axial.get_slices_as_normalized_pixel_arrays(
    shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
print(f"Amount of test images: {len(x_test)}")

y_test = [[int(not(bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in data.test.axial.slices]
y_test = tensorflow.constant(y_test, shape=(len(y_test), 2))


# ---------------------- CALLBACKS ----------------------- #
cb_early_stop = EarlyStopping(monitor='val_mean_squared_error', patience=50, verbose=1)

def lr_exp_decay(epoch, lr):
    k = 0.0069
    # If starting with lr of 1e-3, set k to:
    # 0.00345 to reach lr of 1e-6 at 2000 epochs
    # 0.00690 to reach lr of 1e-6 at 1000 epochs 
    # 0.01365 to reach lr of 1e-6 at 500 epochs
    # 0.03450 to reach lr of 1e-6 at 200 epochs
    # 0.06900 to reach lr of 1e-6 at 100 epochs

    if epoch == 0:
        return lr
    return lr * tensorflow.math.exp(-k)

cb_exp_lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=0)

def lr_drop(epoch, lr):
    if epoch == 10:
        return lr * 1e-1
    return lr

cb_drop_lr_scheduler = LearningRateScheduler(lr_drop, verbose=0)

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
        callbacks=[cb_early_stop, cb_exp_lr_scheduler],
    )
    
    print(f"\n\n---------------------------- SAVING PRE-TRAINED MODEL to {model_path} ----------------------------\n\n")
    autoencoder.save(model_path, save_format='h5')

    default_save_data(autoencoder_history, autoencoder, results, IMAGE_DIM, data.validation.axial)
    

# ------------------- TRANSFER LEARNING ------------------- #
encoder = Model(inputs, encoder)

transfer_learning_classif = Classification_using_transfer_learning(autoencoder, encoder, inputs, x_val, y_val, x_test, y_test)

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
