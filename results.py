import matplotlib.pyplot as plt

# import sklearn.metrics as metrics
import numpy as np
import tensorflow as tf

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

        plt.scatter([i[0] for i in abnormal_predictions],
                    [i[1] for i in abnormal_predictions],
                    c="orange", label="Abnormal")
              
        plt.scatter([i[0] for i in normal_predictions],
                    [i[1] for i in normal_predictions],
                    c="blue", label="Normal", alpha=0.6)
                
        plt.title("Predictions")
        plt.xlabel("Normal slice")
        plt.ylabel("Abnormal slice")
        plt.legend(loc='best')
        plt.savefig(self.__naming("Scatter_plot_classification"))
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


# class Metrics:
#     def __init__(self, true, predict):
#         self.true = true
#         self.predictions = predict
#         self.roc_curve = self.get_roc()
#         self.fpr = self.roc_curve[0]
#         self.tpr = self.roc_curve[1]
#         self.thresholds = self.roc_curve[2]
#         self.auc = self.get_auc()
#         self.confusionmatrix = self.get_confusionmatrix()
#         self.tn = self.confusionmatrix.ravel()[0]
#         self.fp = self.confusionmatrix.ravel()[1]
#         self.fn = self.confusionmatrix.ravel()[2]
#         self.tp = self.confusionmatrix.ravel()[3]
#         self.sensitivity = self.tp / (self.tp + self.fn)
#         self.specificity = self.tn / (self.tn + self.fp)
#         self.f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
#         self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

#     def get_roc(self):
#         return metrics.roc_curve(self.true, self.predictions)

#     def get_auc(self):
#         return metrics.auc(self.fpr, self.tpr)

#     def get_confusionmatrix(self):
#         return metrics.confusion_matrix(self.true, self.predictions)

#     # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

#     def metrics_list(self):
#         return [f"AUC: {self.auc}", f"TP: {self.tp}", ]

#     def print_metrics(self):
#         print("AUC:", self.auc)
#         print("TP:", self.tp)
#         print("FP:", self.fp)
#         print("TN", self.tn)
#         print("FN", self.fn)
#         print("sensitivity", self.sensitivity)
#         print("specificity", self.specificity)
#         print("F1", self.f1)
#         print("Accuracy", self.accuracy)

#     def save_metrics(self):
#         lines =[f"AUC: {self.auc}", f"TP: {self.tp}", f"FP: {self.fp}",
#                 f"TN: {self.tn}", f"FN: {self.fn}", f"Sensitivity: {self.sensitivity}",
#                 f"Specificty: {self.specificity}", f"F1: {self.f1}", f"Accuracy: {self.accuracy}"]
#         with open('metrics.txt', 'w') as f:
#             for line in lines:
#                 f.write(line)

# Metrics(y_true,y_pred).save_metrics()
# a = Classifier(0, 1, y_pred).get_predicted()
# print(a)
# confusion = metrics.confusion_matrix(y_true, y_pred)
#
# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confusion.shape[0]):
#     for j in range(confusion.shape[1]):
#         ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center', size='xx-large')
#
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
#
# fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
# print(metrics.auc(fpr, tpr))
