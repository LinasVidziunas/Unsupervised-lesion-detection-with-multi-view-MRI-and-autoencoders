import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from keras.losses import mse

from processed import View

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
        name = path.join(self.save_in_dir, f"fig_{plotname}")
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
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(self.__naming("MSE_train_vs_val"))
        plt.clf()

    def plot_loss_train_vs_val(self, history):
        """Plot loss for train and validation in the same graph"""

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(self.__naming("loss_train_vs_val"))
        plt.clf()

    def histogram_mse_loss(self, losses_normal,
                         losses_abnormal,
                         name="MSE_loss_hist"):
        """Plot MSE loss for normal and abnormal in the same histogram"""

        plt.clf()
        # Changes done my Mr. Thoresen 18.02.2022
        plt.hist([losses_normal[:]], bins=int(len(losses_normal)), alpha=0.4)
        plt.hist([losses_abnormal[:]], bins=len(losses_abnormal), alpha=0.4)
        plt.xlabel("MSE loss")
        plt.ylabel("No. of slices")
        plt.legend(['Normal', 'Abnormal'], loc='upper left')
        plt.savefig(self.__naming(name))
        plt.clf()

    def histogram_mse_loss_seperate(self, losses_normal,
                                  losses_abnormal, n_bins: int = 7,
                                  name="MSE_loss_hist_seperate"):
        """Plot MSE loss for normal and abnormal in seperate histograms"""

        _, axs = plt.subplots(1, 2, tight_layout=True)

        axs[0].hist([losses_normal[:]], bins=n_bins)
        axs[0].set_title("Normal")
        axs[0].set_xlabel("MSE loss")
        axs[0].set_ylabel("No. of slices")
        axs[1].hist([losses_abnormal[:]], bins=n_bins)
        axs[1].set_title("Abnormal")
        axs[1].set_xlabel("MSE loss")
        axs[1].set_ylabel("No. of slices")
        plt.savefig(self.__naming(name))
        plt.clf()

    def input_vs_reconstructed_images(self, input_images,
                                    reconstructed_images, n: int = 5,
                                    name="input_and_reconstructed_images"):

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

        plt.savefig(self.__naming(name))
        plt.clf()

    def scatter_plot_of_predictions(self, predictions, truth, name="scatter_plot_classification"):
        """
        Saves a scatter plot of predicted class. Arguments need to be categorical,
        where the first value in the list is normal rate and the second is abnormal
        rate.

        :param predictions: tf.tensor categorical prediction ex. [[0.9, 0.1], ...]
        :param truth: categorical truth ex. [[1.0, 0.0], ...]
        """

        abnormal_predictions = []
        normal_predictions = []

        for i, truth in enumerate(truth, start=0):
            if np.array_equal(np.array(truth), np.array([0, 1])):
                abnormal_predictions.append(predictions[i])
            elif np.array_equal(np.array(truth), np.array([1, 0])):
                normal_predictions.append(predictions[i])

        plt.scatter([i[0] for i in normal_predictions],
                    [i[1] for i in normal_predictions],
                    c="blue", label="Normal", alpha=0.6)

        plt.scatter([i[0] for i in abnormal_predictions],
                    [i[1] for i in abnormal_predictions],
                    c="orange", label="Abnormal")

        plt.title("Predictions")
        plt.xlabel("Normal slice")
        plt.ylabel("Abnormal slice")
        plt.legend(loc='best')
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_accuracy(self, thresholds, results_thresholds, name="accuracy_for_thresholds"):
        accuracies = []
        for instance in results_thresholds:
            accuracies.append(instance.get_accuracy())
        plt.plot(thresholds, accuracies, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_sensitivity(self, thresholds, results_thresholds, name="sensitivity_for_thresholds"):
        sensitivities = []
        for instance in results_thresholds:
            sensitivities.append(instance.get_sensitivity())
        plt.plot(thresholds, sensitivities, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Sensitivity")
        plt.grid()
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_specificity(self, thresholds, results_thresholds, name="specificity_for_thresholds"):
        specificities = []
        for instance in results_thresholds:
            specificities.append(instance.get_specificity())
        plt.plot(thresholds, specificities, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Specificity")
        plt.grid()
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_f1(self, thresholds, results_thresholds, name="F1_for_thresholds"):
        f1s = []
        for instance in results_thresholds:
            f1s.append(instance.get_specificity())
        plt.plot(thresholds, f1s, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.grid()
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_roc_curve(self, fpr, tpr, roc_auc, name="ROC_curve"):
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='random estimator')
        display.plot()
        plt.plot([0, 1], [0, 1], 'r:')
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_confusion_matrix(self, confusion_matrix, name="confusion_matrix"):
        display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        display.plot(cmap='Greys')
        plt.savefig(self.__naming(name))
        plt.clf()

    def save_raw_data(self, data, name="raw_data"):
        with open(path.join(
                self.save_in_dir,
                f"{name}{self.timestamp_string()}.raw"), 'w') as f:
            for i, d in enumerate(data):
                if i == len(data) - 1:
                    f.write(f"{d}")
                else:
                    f.write(f"{d},")


# Not the prettiest but removes clutter from main.py
def default_save_data(history, autoencoder, results: ModelResults, IMAGE_DIM,
                    x_val,
                    x_val_abnormal,
                    x_val_normal,
                    mse_keys:list=["mean_squared_error"],
                    val_mse_keys:list=["val_mean_squared_error"],
                    loss_keys:list=["loss"],
                    val_loss_keys:list=["val_loss"],
                    views:list=["axial"]):

    for loss in mse_keys:
        results.save_raw_data(history.history[loss], loss)

    for loss in val_mse_keys:
        results.save_raw_data(history.history[loss], loss)

    for loss in loss_keys:
        results.save_raw_data(history.history[loss], loss)

    for loss in val_loss_keys:
        results.save_raw_data(history.history[loss], loss)

    # Plotting the MSE distrubution of normal slices
    decoded_normal = None

    if len(views) == 1:
        decoded_normal = autoencoder.predict(x_val_normal)

        if isinstance(decoded_normal, tuple):
            decoded_normal = decoded_normal[0]

        loss_normal = mse(decoded_normal.reshape(len(x_val_normal), IMAGE_DIM[0] * IMAGE_DIM[1]),
                          x_val_normal.reshape(len(x_val_normal), IMAGE_DIM[0] * IMAGE_DIM[1]))

        # Saving raw MSE loss of normal slices
        results.save_raw_data(loss_normal, "normal_mse_loss")

        decoded_abnormal = autoencoder.predict(x_val_abnormal)
        if isinstance(decoded_abnormal, tuple):
            decoded_abnormal = decoded_abnormal[0]

        loss_abnormal = mse(decoded_abnormal.reshape(len(x_val_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]),
                            x_val_abnormal.reshape(len(x_val_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))

        # Saving raw MSE loss of abnormal slices
        results.save_raw_data(loss_abnormal, "abnormal_mse_loss")

        # results.plot_mse_train_vs_val(history)
        # results.plot_loss_train_vs_val(history)

        results.histogram_mse_loss(loss_normal, loss_abnormal)
        results.histogram_mse_loss_seperate(loss_normal, loss_abnormal)

        reconstructed_images = autoencoder.predict(x_val)
        if isinstance(reconstructed_images, tuple):
            reconstructed_images = reconstructed_images[0]

        results.input_vs_reconstructed_images(
            [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in x_val],
            [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in reconstructed_images])

    elif len(views) == 3:
        x_val_normal[0] = list(x_val_normal[0].values())
        x_val_normal[1] = list(x_val_normal[1].values())
        x_val_normal[2] = list(x_val_normal[2].values())
        x_val_abnormal[0] = list(x_val_abnormal[0].values())
        x_val_abnormal[1] = list(x_val_abnormal[1].values())
        x_val_abnormal[2] = list(x_val_abnormal[2].values())

        for i, view in enumerate(views):
            decoded_normal = autoencoder.predict(x_val_normal[i])

            loss_normal = mse(decoded_normal[i].reshape(len(x_val_normal[i][i]), IMAGE_DIM[0] * IMAGE_DIM[1]),
                              x_val_normal[i][i].reshape(len(x_val_normal[i][i]), IMAGE_DIM[0] * IMAGE_DIM[1]))

            # Saving raw MSE loss of normal slices
            results.save_raw_data(loss_normal, f"normal_mse_loss_{view}")

            decoded_abnormal = autoencoder.predict(x_val_abnormal[i])

            loss_abnormal = mse(decoded_abnormal[i].reshape(len(x_val_abnormal[i][i]), IMAGE_DIM[0] * IMAGE_DIM[1]),
                                x_val_abnormal[i][i].reshape(len(x_val_abnormal[i][i]), IMAGE_DIM[0] * IMAGE_DIM[1]))

            # Saving raw MSE loss of abnormal slices
            results.save_raw_data(loss_abnormal, f"abnormal_mse_loss_{view}")

            # results.plot_mse_train_vs_val(history)
            # results.plot_loss_train_vs_val(history)

            results.histogram_mse_loss(loss_normal, loss_abnormal, name=f"MSE_loss_hist_{view}")
            results.histogram_mse_loss_seperate(loss_normal, loss_abnormal, name=f"MSE_loss_hist_seperate_{view}")

            reconstructed_images = autoencoder.predict(x_val)[i]

            results.input_vs_reconstructed_images(
                [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in x_val[i]],
                [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in reconstructed_images],
                name=f"input_and_reconstructed_images_{view}")

class Metrics:
    def __init__(self, true, predict):
        self.true = true
        self.predictions = predict
        self.confusionmatrix = self.get_confusionmatrix()
        self.tn = self.confusionmatrix.ravel()[0]
        self.fp = self.confusionmatrix.ravel()[1]
        self.fn = self.confusionmatrix.ravel()[2]
        self.tp = self.confusionmatrix.ravel()[3]
        self.sensitivity = self.tp / (self.tp + self.fn)
        self.specificity = self.tn / (self.tn + self.fp)
        self.f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_confusionmatrix(self):
        return confusion_matrix(self.true, self.predictions)

    def get_results(self):
        return [f"TP: {self.tp}", f"TN: {self.tn}", f"FP: {self.fp}", f"FN: {self.fn}",
                f"Sensitivity: {self.sensitivity}", f"Specificity: {self.specificity}",
                f"F1: {self.f1}", f"Accuracy: {self.accuracy}"]

    def get_f1(self):
        return self.f1

    def get_sensitivity(self):
        return self.sensitivity

    def get_accuracy(self):
        return  self.accuracy

    def get_specificity(self):
        return self.specificity

    def print_metrics(self):
        print("TP:", self.tp)
        print("FP:", self.fp)
        print("TN", self.tn)
        print("FN", self.fn)
        print("sensitivity", self.sensitivity)
        print("specificity", self.specificity)
        print("F1", self.f1)
        print("Accuracy", self.accuracy)

def get_roc(abnormal_losses, normal_losses):
    all_losses = []
    labels = []
    for i in range(len(abnormal_losses)):
        all_losses.append(abnormal_losses[i])
    for i in range(len(normal_losses)):
        all_losses.append(normal_losses[i])

    # Normalize all_losses
    normalized_losses = [el/max(all_losses) for el in all_losses]

    for i in range(len(abnormal_losses)):
        labels.append(1)
    for i in range(len(normal_losses)):
        labels.append(0)

    fpr, tpr, thresholds = roc_curve(labels, normalized_losses)

    # Return "anti-normalized" thresholds
    return fpr, tpr, [el*max(all_losses) for el in thresholds]

def get_auc(fpr, tpr):
    return auc(fpr, tpr)
