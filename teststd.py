import statistics

from sklearn.utils import resample

test_data =["a","a","b","b","c","c"]
test_labels = [0,0,1,1,0,0]
for i in range(5):
    bootstrap_data, bootstrap_labels = resample(test_data, test_labels, replace=True, n_samples=len(test_data),
                                            stratify=test_labels)
    print(bootstrap_data,bootstrap_labels)

# auc_list = [0.5432, 0.5431]
#
# std_auc = statistics.stdev(auc_list)
# print(std_auc)