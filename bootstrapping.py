from keras.losses import mse
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
import statistics


def bootstrapping(model, test_data, test_labels, n_iterations, IMAGE_DIM):
    auc_list = []
    for i in range(n_iterations):
        bootstrap_data, bootstrap_labels = resample(test_data, test_labels, replace=True, n_samples=len(test_data),
                                                    random_state=200, stratify=test_labels)

        predicted = model.predict(bootstrap_data)

        all_losses = mse(predicted.reshape(len(predicted), IMAGE_DIM * IMAGE_DIM),
                         bootstrap_data.reshape(len(bootstrap_data), IMAGE_DIM * IMAGE_DIM))

        normalized_losses = [el / max(all_losses) for el in all_losses]
        fpr, tpr, thresholds = roc_curve(bootstrap_labels, normalized_losses)
        auc = roc_auc_score(fpr, tpr)
        auc_list.append(auc)

    average_auc = sum(auc_list) / len(auc_list)
    std_auc = statistics.stdev(auc_list)

    return average_auc, std_auc
