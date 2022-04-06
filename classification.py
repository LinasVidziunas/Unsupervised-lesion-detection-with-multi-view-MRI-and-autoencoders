import numpy as np

from keras import Model
from keras.losses import CategoricalCrossentropy, mse
from keras.layers import Flatten, Dropout, Dense
from keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam


class Classification:
    def __init__(self, autoencoder: Model, x_val, y_val, x_test, y_test, views):
        self.autoencoder = autoencoder

        # Validation data
        self.x_val = x_val
        self.y_val = y_val

        # Test data
        self.x_test = x_test
        self.y_test = y_test

        self.views = views


class Classification_using_transfer_learning(Classification):
    def __init__(self, autoencoder: Model, encoder: Model, inputs, x_val, y_val, x_test, y_test, views: int = 1):
        super().__init__(autoencoder, x_val, y_val, x_test, y_test, views)

        self.encoder = encoder
        self.inputs = inputs

    def copy_weights(self):
        for i, encoder_layer in enumerate(self.encoder.layers):
            encoder_layer.set_weights(self.autoencoder.layers[i].get_weights())

    def run(self, flatten_layer: bool = False, learning_rate: float = 1e-3, dropout_rate: float = 0.2, batch_size: int = 64, epochs: int = 20):
        # Freeze encoder
        self.encoder.trainable = False

        # New model on top
        x = self.encoder(self.inputs, training=False)

        if flatten_layer:
            x = Flatten()(x)

        x = Dropout(dropout_rate)(x)

        if self.views == 1:
            x = Dense(2, activation='softmax', name="classification")(x)
            self.classif = Model(self.inputs, x)
        else:
            y = []
            for i in range(self.views):
                y.append(Dense(2, activation='softmax', name=f"classification_{i}")(x))

            self.classif = Model(self.inputs, y)

        self.classif.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=[CategoricalCrossentropy() for _ in range(self.views)],
            metrics=[CategoricalAccuracy()])

        self.classif.summary()

        return self.classif.fit(
            self.x_val,
            self.y_val,
            batch_size=batch_size,
            epochs=epochs)

    def fine_tune(self, learning_rate: float = 1e-5, batch_size: int = 64, epochs: int = 10, num_layers: int = 0):
        # Unfreeze the last 'num_layers' in the encoder
        if num_layers != 0:
            for layer_index in range(len(self.encoder.layers) - 1, -1 + len(self.encoder.layers) - num_layers, -1):
                self.encoder.layers[layer_index].trainable = True
        elif num_layers == 0:
            self.encoder.trainable = True

        self.classif.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=[CategoricalCrossentropy() for _ in range(self.views)],
            metrics=[CategoricalAccuracy()])

        self.classif.summary()

        return self.classif.fit(
            self.x_val,
            self.y_val,
            batch_size=batch_size,
            epochs=epochs)


class IQR_method(Classification):
    def __init__(self, autoencoder: Model, x_val, y_val, x_test, y_test, image_dim, view, views):
        super().__init__(autoencoder, x_val, y_val, x_test, y_test, views)

        validation_decoded = self.autoencoder.predict(x_val)
        if views == 1 and isinstance(validation_decoded, tuple):
            validation_decoded = validation_decoded[0]

        test_decoded = self.autoencoder.predict(x_test)
        if views == 1 and isinstance(test_decoded, tuple):
            test_decoded = test_decoded[0]

        if views != 1:
            validation_decoded = validation_decoded[view]
            x_val = x_val[view]
            test_decoded = test_decoded[view]
            x_test = x_test[view]

        self.validation_losses = mse(
            validation_decoded.reshape(len(validation_decoded), image_dim[0] * image_dim[1]),
            x_val.reshape(len(x_val), image_dim[0] * image_dim[1]))

        self.test_losses = mse(
            test_decoded.reshape(len(test_decoded), image_dim[0] * image_dim[1]),
            x_test.reshape(len(x_test), image_dim[0] * image_dim[1]))

    def obtain_threshold(self, K: float=1.5):
        q3, q1 = np.percentile(self.validation_losses, [75, 25])
        iqr = q3 - q1
        threshold = q3 + K * iqr
        return threshold

    def classify(self, threshold: float):
        predicted = []
        for loss in self.test_losses:
            if loss < threshold:
                predicted.append(0)
            elif loss >= threshold:
                predicted.append(1)
        return predicted

