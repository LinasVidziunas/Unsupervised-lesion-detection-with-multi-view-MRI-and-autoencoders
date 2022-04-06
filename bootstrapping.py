from keras.losses import mse
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import statistics

def bootstrapping_multiview_mse(model, test_data, test_labels, n_iterations, IMAGE_DIM):
    aucs = {"axial": [], "coronal": [], "sagittal": []}

    for iteration in range(n_iterations):
        print(f"Iteration {iteration}")

        for i, view in enumerate(aucs.keys()):
            axial, coronal, sagittal, i_labels = resample(test_data[0], test_data[1], test_data[2], test_labels[i],
                                                          replace=True, n_samples=len(test_data[i]),
                                                          stratify=test_labels[i])
            i_labels = [el[1] for el in i_labels]
            the_view = [axial, coronal, sagittal]

            predicted = model.predict([axial, coronal, sagittal])[i]

            losses = mse(predicted.reshape(len(predicted), IMAGE_DIM * IMAGE_DIM),
                         the_view[i].reshape(len(the_view[i]), IMAGE_DIM * IMAGE_DIM))

            normalized_axial_losses = [el / max(losses) for el in losses]

            fpr, tpr, _ = roc_curve(i_labels, normalized_axial_losses)
            auc_score = auc(fpr, tpr)

            aucs[view].append(auc_score)       

    average_auc = {
        "axial": sum(aucs['axial']) / len(aucs['axial']),
        "coronal": sum(aucs['coronal']) / len(aucs['coronal']),
        "sagittal": sum(aucs['sagittal']) / len(aucs['sagittal'])
    }

    std_auc = {
        "axial": statistics.stdev(aucs['axial']),
        "coronal": statistics.stdev(aucs['coronal']),
        "sagittal": statistics.stdev(aucs['sagittal'])
    }

    return average_auc, std_auc


def bootstrapping_multiview_TL(model, test_data, test_labels, n_iterations):
    aucs = {"axial": [], "coronal": [], "sagittal": []}

    for iteration in range(n_iterations):
        print("Iteration", str(iteration))

        for i, view in enumerate(aucs.keys()):
            axial, coronal, sagittal, i_labels = resample(test_data[0], test_data[1], test_data[2], test_labels[i],
                                                          replace=True, n_samples=len(test_data[i]),
                                                          stratify=test_labels[i])

            i_labels = [el[1] for el in i_labels]
            the_view = [axial, coronal, sagittal]

            predictions = model.classif.predict(the_view)[i]

            predictions = [x[1] for x in predictions]
            normalized_predictions = [el / max(predictions) for el in predictions]

            fpr, tpr, _ = roc_curve(i_labels, normalized_predictions)
            auc_score = auc(fpr, tpr)

            aucs[view].append(auc_score)       

    average_auc = {
        "axial": sum(aucs['axial']) / len(aucs['axial']),
        "coronal": sum(aucs['coronal']) / len(aucs['coronal']),
        "sagittal": sum(aucs['sagittal']) / len(aucs['sagittal'])
    }

    std_auc = {
        "axial": statistics.stdev(aucs['axial']),
        "coronal": statistics.stdev(aucs['coronal']),
        "sagittal": statistics.stdev(aucs['sagittal'])
    }


    return average_auc, std_auc

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
        fpr, tpr, _ = roc_curve(bootstrap_labels, normalized_losses)
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
        fpr, tpr, _ = roc_curve(bootstrap_labels, normalized_predictions)
        auc_score = auc(fpr, tpr)
        auc_list.append(auc_score)

    average_auc = sum(auc_list) / len(auc_list)
    std_auc = statistics.stdev(auc_list)

    return average_auc, std_auc
