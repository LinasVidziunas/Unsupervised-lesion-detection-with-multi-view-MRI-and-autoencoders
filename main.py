from PatientDataExtractor.dataextractor import Column, PatientDataExtractor, keep_columns
from patient_data_preprocessing import Patient

from os import listdir, path
from random import shuffle


def split_set_by_patients(patients, train_amount: int, validation_amount: int,
                          test_amount: int):

    if len(patients) != train_amount + validation_amount + test_amount:
        raise ValueError(
            "Length of patients is not equal to the total amount of set amounts"
        )

    shuffle(patients)

    train_set = patients[0:train_amount]
    validation_set = patients[train_amount:train_amount + validation_amount]
    test_set = patients[train_amount + validation_amount:train_amount +
                        validation_amount + test_amount]

    return {"train": train_set, "validation": validation_set, "test": test_set}


data = PatientDataExtractor("ProstateX-Images-Train.csv")
abnormal_list = keep_columns(
    data.filter_by_column(Column.DCMSerDescr,
                          ["t2_tse_sag", "t2_tse_cor", "t2_tse_tra"]),
    [Column.ProxID, Column.DCMSerNum, Column.ijk])

for i, patient in enumerate(abnormal_list):
    # from ijk, keeping only k (the slice no)
    abnormal_list[i][1] = str(abnormal_list[i][1]).split(" ")[2]

patients = []
PROSTATEx_path = "PROSTATEx"
for i, folder_path in enumerate(listdir(PROSTATEx_path)):
    patients.append(
        Patient(path.join(PROSTATEx_path, folder_path), abnormal_list))
    # patients[i].extract(path.join("test", folder_path))

sets = split_set_by_patients(patients, 1, 1, 0)

# After splitting set, extract patient dicoms to new directory
for set_name, patients in sets.items():
    for i, patient in enumerate(patients):
        patient.extract(path.join("sets", set_name))
