from keras.callbacks import Callback
from keras.losses import mse
from results import ModelResults

from os import path

class ResultsCallback(Callback):
    def __init__(self, model_name, image_dim, val_view, save_at_epochs=[25, 50, 100, 200, 300]):
        self.model_name = model_name
        self.image_dim = image_dim
        self.val_view = val_view
        self.save_at_epochs = save_at_epochs

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch in self.save_at_epochs:
            results = ModelResults(path.join(f"{self.model_name}", path.join("epoch", str(epoch))))
            print(f"Saving results on epoch {epoch} of training")

            x_val_abnormal = self.val_view.get_abnormal_slices_as_normalized_pixel_arrays(
                shape=(self.image_dim[0], self.image_dim[1]))
            x_val_normal = self.val_view.get_normal_slices_as_normalized_pixel_arrays(
                shape=(self.image_dim[0], self.image_dim[1]))

            # Plotting the MSE distrubution of normal slices
            decoded_normal = self.model.predict(x_val_normal)
            loss_normal = mse(decoded_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]),
                              x_val_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]))

            # Saving raw MSE loss of normal slices
            results.save_raw_data(loss_normal, "normal_mse_loss")

            decoded_abnormal = self.model.predict(x_val_abnormal)
            loss_abnormal = mse(
                decoded_abnormal.reshape(len(x_val_abnormal), self.image_dim[0] * self.image_dim[1]),
                x_val_abnormal.reshape(len(x_val_abnormal), self.image_dim[0] * self.image_dim[1]))

            # Saving raw MSE loss of abnormal slices
            results.save_raw_data(loss_abnormal, "abnormal_mse_loss")

            results.histogram_mse_loss(loss_normal, loss_abnormal)
            results.histogram_mse_loss_seperate(loss_normal, loss_abnormal)
            
            x_val = self.val_view.get_slices_as_normalized_pixel_arrays(
                shape=(self.image_dim[0], self.image_dim[1]))

            reconstructed_images = self.model.predict(x_val)

            results.input_vs_reconstructed_images(
                [el.reshape(self.image_dim[0], self.image_dim[1]) for el in x_val],
                [el.reshape(self.image_dim[0], self.image_dim[1]) for el in reconstructed_images]
            )
