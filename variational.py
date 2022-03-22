import tensorflow as tf
from keras import Model
from keras.losses import binary_crossentropy
from keras.layers import Layer
from keras.metrics import MeanSquaredError, Mean

# Credits:
# Sampling layer and train_step functions are directly obtained from 
# https://keras.io/examples/generative/vae/


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    """A VAE wrapper for ae based on VGG16"""
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.mse_loss_tracker = MeanSquaredError(name="mean_squared_error")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var, _ = self(x, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(tf.expand_dims(y, -1), reconstructed), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_loss_tracker.update_state(y, reconstructed)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mean_squared_error": self.mse_loss_tracker.result()
        }
