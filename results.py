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
                           losses_abnormal):
        """Plot MSE loss for normal and abnormal in the same histogram"""
        # Changes done my Mr. Thoresen 18.02.2022
        plt.hist([losses_normal[:]], bins=int(len(losses_normal)), alpha=0.4)
        plt.hist([losses_abnormal[:]], bins=len(losses_abnormal), alpha=0.4)
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

    def scatter_plot_of_predictions(self, predictions, truth):
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
        plt.savefig(self.__naming("Scatter_plot_classification"))
        plt.clf()

    def plot_accuracy(self, thresholds, results_thresholds):
        accuracies = []
        for instance in results_thresholds:
            accuracies.append(instance.get_accuracy())
        plt.plot(thresholds, accuracies, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.savefig(self.__naming("Accuracy_for_thresholds"))
        plt.clf()

    def plot_sensitivity(self, thresholds, results_thresholds):
        sensitivities = []
        for instance in results_thresholds:
            sensitivities.append(instance.get_sensitivity())
        plt.plot(thresholds, sensitivities, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Sensitivity")
        plt.grid()
        plt.savefig(self.__naming("Sensitivity_for_thresholds"))
        plt.clf()

    def plot_specificity(self, thresholds, results_thresholds):
        specificities = []
        for instance in results_thresholds:
            specificities.append(instance.get_specificity())
        plt.plot(thresholds, specificities, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("Specificity")
        plt.grid()
        plt.savefig(self.__naming("Specificity_for_thresholds"))
        plt.clf()

    def plot_f1(self, thresholds, results_thresholds):
        f1s = []
        for instance in results_thresholds:
            f1s.append(instance.get_specificity())
        plt.plot(thresholds, f1s, "o-")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.grid()
        plt.savefig(self.__naming("F1_for_thresholds"))
        plt.clf()

    def plot_roc_curve(self, fpr, tpr, roc_auc, name="ROC_curve"):
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='random estimator')
        display.plot()
        plt.plot([0, 1], [0, 1], 'r:')
        plt.savefig(self.__naming(name))
        plt.clf()

    def plot_confusion_matrix(self, confusion_matrix):
        display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        display.plot(cmap='Greys')
        plt.savefig(self.__naming("confusion_matrix"))
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
def default_save_data(history, autoencoder, results: ModelResults, IMAGE_DIM, val_view: View):
    results.save_raw_data(history.history['mean_squared_error'], "mse_per_epoch")
    # results.save_raw_data(history.history['val_mean_squared_error'], "val_mse_per_epoch")
    results.save_raw_data(history.history['loss'], "loss_epoch")
    # results.save_raw_data(history.history['val_loss'], "val_loss_epoch")

    val_view.get_abnormal_slices_as_normalized_pixel_arrays

    x_val_abnormal = val_view.get_abnormal_slices_as_normalized_pixel_arrays(
        shape=(IMAGE_DIM[0], IMAGE_DIM[1]))
    x_val_normal = val_view.get_normal_slices_as_normalized_pixel_arrays(
        shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

    # Plotting the MSE distrubution of normal slices
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

    loss_abnormal = mse(
        decoded_abnormal.reshape(len(x_val_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]),
        x_val_abnormal.reshape(len(x_val_abnormal), IMAGE_DIM[0] * IMAGE_DIM[1]))

    # Saving raw MSE loss of abnormal slices
    results.save_raw_data(loss_abnormal, "abnormal_mse_loss")

    # results.plot_mse_train_vs_val(history)
    # results.plot_loss_train_vs_val(history)

    results.histogram_mse_loss(loss_normal, loss_abnormal)
    results.histogram_mse_loss_seperate(loss_normal, loss_abnormal)

    x_val = val_view.get_slices_as_normalized_pixel_arrays(
        shape=(IMAGE_DIM[0], IMAGE_DIM[1]))

    reconstructed_images = autoencoder.predict(x_val)
    if isinstance(reconstructed_images, tuple):
        reconstructed_images = reconstructed_images[0]

    results.input_vs_reconstructed_images(
        [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in x_val],
        [el.reshape(IMAGE_DIM[0], IMAGE_DIM[1]) for el in reconstructed_images]
    )

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

# y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


# def get_iqr(reconstruction_error_normal):
#     q3, q1 = np.percentile(reconstruction_error_normal, [75, 25])
#     median = np.mean(reconstruction_error_normal)
#     iqr = q3 - q1
#     return median, iqr


# class Classifier:
#     def __init__(self, median, iqr, data):
#         self.median = median
#         self.iqr = iqr
#         self.data = data
#         self.predicted = []

#     def get_predicted(self):
#         for i in self.data:
#             if i > (self.median + self.iqr):
#                 self.predicted.append(1)
#             else:
#                 self.predicted.append(0)
#         return self.predicted

