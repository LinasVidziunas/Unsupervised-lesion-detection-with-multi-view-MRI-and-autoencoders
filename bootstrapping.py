from keras.losses import mse
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import statistics


def bootstrapping_mse(model, test_data, test_labels, n_iterations, IMAGE_DIM):
    auc_list = []
    for i in range(n_iterations):
        print("Iteration", str(i))
        bootstrap_data, bootstrap_labels = resample(test_data, test_labels, replace=True, n_samples=len(test_data),
                                                     stratify=test_labels)

        predicted = model.predict(bootstrap_data)

        all_losses = mse(predicted.reshape(len(predicted), IMAGE_DIM * IMAGE_DIM),
                         bootstrap_data.reshape(len(bootstrap_data), IMAGE_DIM * IMAGE_DIM))

        normalized_losses = [el / max(all_losses) for el in all_losses]
        fpr, tpr, thresholds = roc_curve(bootstrap_labels, normalized_losses)
        auc_score = auc(fpr, tpr)
        auc_list.append(auc_score)

    average_auc = sum(auc_list) / len(auc_list)
    std_auc = statistics.stdev(auc_list)

    return average_auc, std_auc

def bootstrapping_TL(model, test_data, test_labels, n_iterations):
    auc_list = []
    for i in range(n_iterations):
        print("Iteration", str(i))
        bootstrap_data, bootstrap_labels = resample(test_data, test_labels, replace=True, n_samples=len(test_data),
                                                     stratify=test_labels)


        predictions = model.classif.predict(bootstrap_data)
        predictions = [x[1] for x in predictions]


        normalized_predictions = [el / max(predictions) for el in predictions]
        fpr, tpr, thresholds = roc_curve(bootstrap_labels, normalized_predictions)
        auc_score = auc(fpr, tpr)
        auc_list.append(auc_score)

    average_auc = sum(auc_list) / len(auc_list)
    std_auc = statistics.stdev(auc_list)

    return average_auc, std_auc
