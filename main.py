from PatientDataExtractor.dataextractor import Column, PatientDataExtractor, keep_columns
from patient_data_preprocessing import PatientDataPreprocessing

from os import listdir, path

data = PatientDataExtractor("ProstateX-Images-Train.csv")
abnormal_list = keep_columns(data.filter_by_column(Column.DCMSerDescr, ["t2_tse_sag", "t2_tse_cor", "t2_tse_tra"]),
                                      [Column.ProxID,
                                       Column.DCMSerNum,
                                       Column.ijk])


for i, patient in enumerate(abnormal_list):
    # from ijk, keeping only k (the slice no)
    abnormal_list[i][1] = str(abnormal_list[i][1]).split(" ")[2]

patients = []
PROSTATEx_path="PROSTATEx"
for i, folder_path in enumerate(listdir(PROSTATEx_path)):
    patients.append(PatientDataPreprocessing(path.join(PROSTATEx_path, folder_path)))
    patients[i].extract(path.join("test", folder_path), abnormal_list)

