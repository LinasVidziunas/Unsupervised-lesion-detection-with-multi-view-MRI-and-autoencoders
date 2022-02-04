import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]

class Metrics:
    def __init__(self, true, predict):
        self.true = true
        self.predictions = predict
        self.roc_curve = self.get_roc()
        self.fpr = self.roc_curve[0]
        self.tpr = self.roc_curve[1]
        self.thresholds = self.roc_curve[2]
        self.auc = self.get_auc()
        self.confusionmatrix = self.get_confusionmatrix()
        self.tn = self.confusionmatrix.ravel()[0]
        self.fp = self.confusionmatrix.ravel()[1]
        self.fn = self.confusionmatrix.ravel()[2]
        self.tp = self.confusionmatrix.ravel()[3]
        self.sensitivity = self.tp/(self.tp+self.fn)
        self.specificity = self.tn/(self.tn+self.fp)
        self.f1 = 2*self.tp/(2*self.tp+self.fp+self.fn)
        self.accuracy = (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)


    def get_roc(self):
        return metrics.roc_curve(self.true, self.predictions)

    def get_auc(self):
        return metrics.auc(self.fpr, self.tpr)

    def get_confusionmatrix(self):
        return metrics.confusion_matrix(self.true, self.predictions)
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

    def metrics_list(self):
        return [f"AUC: {self.auc}", f"TP: {self.tp}", ]

    def print_metrics(self):
        print("AUC:", self.auc)
        print("TP:", self.tp)
        print("FP:", self.fp)
        print("TN", self.tn)
        print("FN", self.fn)
        print("sensitivity", self.sensitivity)
        print("specificity", self.specificity)
        print("F1", self.f1)
        print("Accuracy", self.accuracy)

    def save_metrics(self):
        line = f"AUC: {self.auc}", f""
        with open('readme.txt', 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n'



Metrics(y_true, y_pred).print_metrics()

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