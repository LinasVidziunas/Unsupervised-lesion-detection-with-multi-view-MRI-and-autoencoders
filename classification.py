from keras import Model
from keras.losses import CategoricalCrossentropy
from keras.layers import Flatten, Dropout, Dense
from keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam


class Classification:
    def __init__(self, autoencoder: Model, x_val, y_val, x_test, y_test):
        self.autoencoder = autoencoder

        # Validation data
        self.x_val = x_val
        self.y_val = y_val

        # Test data
        self.x_test = x_test
        self.y_test = y_test


class Classification_using_transfer_learning(Classification):
    def __init__(self, autoencoder: Model, encoder: Model, inputs, x_val, y_val, x_test, y_test):
        super().__init__(autoencoder, x_val, y_val, x_test, y_test)

        self.encoder = encoder
        self.inputs = inputs

    def copy_weights(self):
        for i, encoder_layer in enumerate(self.encoder.layers):
            encoder_layer.set_weights(self.autoencoder.layers[i].get_weights())

    def run(self, flatten_layer: bool = False, learning_rate: float = 1e-3, dropout_rate: float = 0.2, batch_size: int = 64, epochs: int = 20):
        # Freeze encoder
        self.encoder.trainabe = False

        # New model on top
        x = self.encoder(self.inputs, training=False)

        if flatten_layer:
            x = Flatten()(x)

        x = Dropout(dropout_rate)(x)
        x = Dense(2, activation='softmax', name="classification")(x)
        self.classif = Model(self.inputs, x)

        self.classif.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()])

        self.classif.summary()

        return self.classif.fit(
            self.x_val,
            self.y_val,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test))

    def fine_tune(self, learning_rate: float = 1e-5, batch_size: int = 64, epochs: int = 10, num_layers: int = 5):
        # Unfreeze the last 'num_layers' in the encoder
        for layer_index in range(len(self.encoder.layers) - 1, -1 + len(self.encoder.layers) - num_layers, -1):
            self.encoder.layers[layer_index].trainable = True

        self.classif.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()])

        self.classif.summary()

        return self.classif.fit(
            self.x_val,
            self.y_val,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test))
