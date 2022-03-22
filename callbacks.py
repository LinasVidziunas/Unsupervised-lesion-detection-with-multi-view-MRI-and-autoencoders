from keras.callbacks import Callback
from keras.losses import mse
from results import ModelResults
from results import get_roc, get_auc
import matplotlib.pyplot as plt

from os import path

class AUCresult():
    def __init__(self, epoch, tpr, fpr):
        self.epoch = epoch
        self.tpr = tpr
        self.fpr = fpr
        self.auc_score = get_auc(fpr, tpr)


class AUCcallback(Callback):
    def __init__(self, results: ModelResults, val_view, image_dim, every_x_epochs=5):
        self.results = results
        self.val_view = val_view
        self.image_dim = image_dim
        self.every_x_epochs = every_x_epochs

        self.auc_results = []

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.every_x_epochs != 0:
            return

        x_val_abnormal = self.val_view.get_abnormal_slices_as_normalized_pixel_arrays(
            shape=(self.image_dim[0], self.image_dim[1]))
        x_val_normal = self.val_view.get_normal_slices_as_normalized_pixel_arrays(
            shape=(self.image_dim[0], self.image_dim[1]))


        decoded_normal = self.model.predict(x_val_normal)
        if isinstance(decoded_normal, tuple):
            decoded_normal = decoded_normal[0]

        loss_normal = mse(decoded_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]),
                          x_val_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]))

        decoded_abnormal = self.model.predict(x_val_abnormal)
        if isinstance(decoded_abnormal, tuple):
            decoded_abnormal = decoded_abnormal[0]

        loss_abnormal = mse(
            decoded_abnormal.reshape(len(x_val_abnormal), self.image_dim[0] * self.image_dim[1]),
            x_val_abnormal.reshape(len(x_val_abnormal), self.image_dim[0] * self.image_dim[1]))

        fpr, tpr, _ = get_roc(loss_abnormal, loss_normal)
        self.auc_results.append(AUCresult(epoch, tpr, fpr))

    def on_train_end(self, logs=None):
        plt.plot([res.epoch for res in self.auc_results], [res.auc_score for res in self.auc_results])
        plt.title(f"AUC per {self.every_x_epochs} epoch(s)")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.savefig(
            path.join(
                self.results.save_in_dir,
                f"fig_auc_per_{self.every_x_epochs}_epochs-{self.results.timestamp_string()}.png"))
        plt.clf()


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

            decoded_normal = self.model.predict(x_val_normal)
            if isinstance(decoded_normal, tuple):
                decoded_normal = decoded_normal[0]

            loss_normal = mse(decoded_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]),
                              x_val_normal.reshape(len(x_val_normal), self.image_dim[0] * self.image_dim[1]))

            # Saving raw MSE loss of normal slices
            results.save_raw_data(loss_normal, "normal_mse_loss")

            decoded_abnormal = self.model.predict(x_val_abnormal)
            if isinstance(decoded_abnormal, tuple):
                decoded_abnormal = decoded_abnormal[0]

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
            if isinstance(reconstructed_images, tuple):
                reconstructed_images = reconstructed_images[0]

            results.input_vs_reconstructed_images(
                [el.reshape(self.image_dim[0], self.image_dim[1]) for el in x_val],
                [el.reshape(self.image_dim[0], self.image_dim[1]) for el in reconstructed_images]
            )


            # ---------------------------- Saving validation roc and auc -----------------------------#
            fpr, tpr, _ = get_roc(loss_abnormal, loss_normal)
            auc_score = get_auc(fpr, tpr)

            results.plot_roc_curve(fpr, tpr, auc_score)
