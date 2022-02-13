import matplotlib.pyplot as plt
from datetime import datetime

from os import path, makedirs


class ModelResults:
    def __init__(
            self,
            model_name: str = "default",
            timestamp: bool = True):
        """
        Easier plotting without having to worry about naming, timestamping,
        and so on.

        :param model_name: Subdirectory of results where to save figures (Default: "default")
        :param timestamp: Boolean value whether to timestamp figure names
        :type timestamp: bool
        :return: Instance of ModelPlotting,
        from where you can call plotting functions.
        """

        # self.history = history
        self.save_in_dir = path.join('results', model_name)

        if not path.isdir(self.save_in_dir):
            makedirs(self.save_in_dir, exist_ok=True)

        self.timestamp = timestamp

    def timestamp_string(self):
        today = datetime.today()
        return f"{today.day}{today.month}{today.year}-{today.hour}{today.minute}"

    def __naming(self, plotname: str):
        name = f"{self.save_in_dir}/fig_{plotname}"
        if self.timestamp:
            name += f"-{self.timestamp_string()}"
        name += ".png"
        return name

    def plot_mse_train_vs_val(self, history):
        """Plot MSE loss for train and validation in the same graph"""

        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model MSE loss')
        plt.ylabel('MSE loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.__naming("MSE_train_vs_val"))
        plt.clf()

    def plot_loss_train_vs_val(self, history):
        """Plot loss for train and validation in the same graph"""

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.__naming("loss_train_vs_val"))
        plt.clf()

    def histogram_mse_loss(self, losses_normal,
                           losses_abnormal, n_bins: int = 15):
        """Plot MSE loss for normal and abnormal in the same histogram"""

        plt.hist([losses_normal[:], losses_abnormal[:]], bins=n_bins)
        plt.xlabel("MSE loss")
        plt.ylabel("No. of slices")
        plt.legend(['Normal', 'Abnormal'], loc='upper left')
        plt.savefig(self.__naming("MSE_loss_hist"))
        plt.clf()

    def histogram_mse_loss_seperate(self, losses_normal,
                                    losses_abnormal, n_bins: int = 7):
        """Plot MSE loss for normal and abnormal in seperate histograms"""

        fig, axs = plt.subplots(1, 2, tight_layout=True)

        axs[0].hist([losses_normal[:]], bins=n_bins)
        axs[0].set_title("Normal")
        axs[0].set_xlabel("MSE loss")
        axs[0].set_ylabel("No. of slices")
        axs[1].hist([losses_abnormal[:]], bins=n_bins)
        axs[1].set_title("Abnormal")
        axs[1].set_xlabel("MSE loss")
        axs[1].set_ylabel("No. of slices")
        plt.savefig(self.__naming("MSE_loss_hist_seperate"))
        plt.clf()

    def input_vs_reconstructed_images(self, input_images,
                                      reconstructed_images, n: int = 10):

        plt.figure(figsize=(20, 4))

        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(input_images[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstructed
            ax = plt.subplot(2, n, i + n)
            plt.imshow(reconstructed_images[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(self.__naming("input_and_reconstructed_images"))
        plt.clf()
