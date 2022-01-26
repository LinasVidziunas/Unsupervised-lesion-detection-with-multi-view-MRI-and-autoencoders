from Datapreprocessing.dataextractor import Column, PatientDataExtractor
from Datapreprocessing.dataextractor import keep_columns
from Datapreprocessing.splitset import split_set_by_patients
from Datapreprocessing.patient import Patient

from os import listdir, path

# read_csv file
data = PatientDataExtractor("Datapreprocessing/ProstateX-Images-Train.csv")
abnormal_list = keep_columns(
    data.filter_by_column(Column.DCMSerDescr,
                          ["t2_tse_sag", "t2_tse_cor", "t2_tse_tra"]),
    [Column.ProxID, Column.DCMSerNum, Column.ijk])

# from ijk, keeping only k (the slice no)
for i, patient in enumerate(abnormal_list):
    abnormal_list[i][1] = str(abnormal_list[i][1]).split(" ")[2]

patients = []
PROSTATEx_path = "Datapreprocessing/PROSTATEx"
for i, folder_path in enumerate(listdir(PROSTATEx_path)):
    patients.append(
        Patient(path.join(PROSTATEx_path, folder_path), abnormal_list))
    # patients[i].extract(path.join("test", folder_path))

sets = split_set_by_patients(patients, 1, 1, 0)

# After splitting set, extract patient dicoms to new directory
for set_name, patients in sets.items():
    for i, patient in enumerate(patients):
        patient.extract(path.join("sets", set_name))
