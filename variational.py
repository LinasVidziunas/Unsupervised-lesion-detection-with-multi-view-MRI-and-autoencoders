import tensorflow as tf
from keras import Model
from keras.losses import binary_crossentropy
from keras.layers import Layer, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dense, Flatten, Reshape
from keras.metrics import MeanSquaredError, Mean


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE_UNET(Model):
    """A wrapper for a VAE UNET model"""

    def __init__(self,
               inputs,
               encoder_filters=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024],
               decoder_filters=[512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64],
               latent_dim: int = 576, **kwargs):
        super(VAE_UNET, self).__init__(**kwargs)
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.latent_dim = latent_dim

        self.outputs, self.z_mean, self.z_log_var, self.z  = self._model(inputs)
        self.model = Model(inputs, [self.outputs, self.z_mean, self.z_log_var, self.z], name="VAE_UNET")

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.mse_loss_tracker = MeanSquaredError(name="mean_squared_error")


    @property
    def layers(self):
        return self.model.layers


    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker,
           self.kl_loss_tracker, self.mse_loss_tracker]


    def _model(self, inputs,
              encoder_filters=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 128],
              decoder_filters=[512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]
              ):
        c1 = Conv2D(encoder_filters[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1skip = Conv2D(encoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

        p1 = MaxPooling2D((2, 2))(c1skip)
        c2 = Conv2D(encoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2skip = Conv2D(encoder_filters[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        
        p2 = MaxPooling2D((2, 2))(c2skip)
        c3 = Conv2D(encoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3skip = Conv2D(encoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        
        p3 = MaxPooling2D((2, 2))(c3skip)
        c4 = Conv2D(encoder_filters[6], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4skip = Conv2D(encoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        
        p4 = MaxPooling2D(pool_size=(2, 2))(c4skip)
        c5 = Conv2D(encoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Conv2D(encoder_filters[9], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        flatten = Flatten()(c5)

        # Change name
        flatten = Dense(3000)(flatten)

        z_mean = Dense(self.latent_dim, name="z_mean")(flatten)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(flatten)
        z = Sampling()([z_mean, z_log_var])

        z = Dense(73728)(z)

        reshape = Reshape((24, 24, 128))(z)

        u6 = Conv2DTranspose(decoder_filters[0], (2, 2), strides=(2, 2), padding='same')(reshape)
        u6 = concatenate([u6, c4skip])
        c6 = Conv2D(decoder_filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Conv2D(decoder_filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(decoder_filters[3], (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3skip])
        c7 = Conv2D(decoder_filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Conv2D(decoder_filters[5], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(decoder_filters[6], (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2skip])
        c8 = Conv2D(decoder_filters[7], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Conv2D(decoder_filters[8], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(decoder_filters[9], (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1skip])
        c9 = Conv2D(decoder_filters[10], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Conv2D(decoder_filters[11], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        output = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same')(c9)
        return output, z_mean, z_log_var, z


    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var, _ = self.model(data)
            data = tf.expand_dims(data, -1)

            reconstruction_loss = binary_crossentropy(data, reconstructed)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_loss_tracker.update_state(data, reconstructed)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mean_squared_error": self.mse_loss_tracker.result()
        }


    def predict(self, images: list):
        return self.model.predict(images)[0]


    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False):
        return self.model.summary(line_length, positions, print_fn, expand_nested, show_trainable)
