import matplotlib.pyplot as plt
from datetime import datetime


class ModelPlotting:
    def __init__(
            self,
            history,
            save_in_dir: str = "",
            timestamp: bool = True):
        """
        Easier plotting without having to worry about naming, timestamping,
        and so on.

        :param history: The history that Model.fit returns
        :param save_in_dir: Directory where to save figures (Default: "")
        :param timestamp: Boolean value whether to timestamp figure names
        :type timestamp: bool
        :return: Instance of ModelPlotting,
        from where you can call plotting functions.
        """

        self.history = history
        self.save_in_dir = save_in_dir
        self.timestamp = timestamp

    def __timestamp_string(self):
        today = datetime.today()
        return f"{today.day}-{today.month}-{today.hour}-{today.minute}"

    def __naming(self, plotname: str):
        name = f"{self.save_in_dir}/fig_{plotname}"
        if self.timestamp:
            name += f"-{self.__timestamp_string()}"
        name += ".png"
        return name

    def plot_mae_train_vs_val(self):
        """Plot MAE loss for train and validation in the same graph"""

        plt.plot(self.history.history['mae'])
        plt.plot(self.history.history['val_mae'])
        plt.title('Model MAE loss')
        plt.ylabel('MAE loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.__naming("MAE_train_vs_val"))
        plt.clf()

    def plot_loss_train_vs_val(self):
        """Plot loss for train and validation in the same graph"""

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.__naming("loss_train_vs_val"))
        plt.clf()

    def histogram_mae_loss(self, losses_normal,
                           losses_abnormal, n_bins: int = 15):
        """Plot MAE loss for normal and abnormal in the same histogram"""

        plt.hist([losses_normal[:], losses_abnormal[:]], bins=n_bins)
        plt.xlabel("MAE loss")
        plt.ylabel("No. of slices")
        plt.legend(['Normal', 'Abnormal'], loc='upper left')
        plt.savefig(self.__naming("MAE_loss_hist"))
        plt.clf()

    def histogram_mae_loss_seperate(self, losses_normal,
                                    losses_abnormal, n_bins: int = 7):
        """Plot MAE loss for normal and abnormal in seperate histograms"""

        fig, axs = plt.subplots(1, 2, tight_layout=True)

        axs[0].hist([losses_normal[:]], bins=n_bins)
        axs[0].set_title("Normal")
        axs[0].set_xlabel("MAE loss")
        axs[0].set_ylabel("No. of slices")
        axs[1].hist([losses_abnormal[:]], bins=n_bins)
        axs[1].set_title("Abnormal")
        axs[1].set_xlabel("MAE loss")
        axs[1].set_ylabel("No. of slices")
        plt.savefig(self.__naming("MAE_loss_hist_seperate"))
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


# Not how one does it, but keeping the code to remind myself to do this
# def save_summary(
#         summary,
#         save_as=f"model_summary-{timestamp_string()}.png"):

#     with open(save_as, 'w') as file:
#         file.write(summary)
