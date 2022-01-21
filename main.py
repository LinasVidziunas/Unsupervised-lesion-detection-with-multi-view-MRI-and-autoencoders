from dataextractor import Column, PatientDataExtractor, keep_columns
from patient_data_preprocessing import PatientDataPreprocessing

from os import listdir, path

data = PatientDataExtractor("/home/linas/Downloads/ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv")
abnormal_list = keep_columns(data.filter_by_column(Column.DCMSerDescr, ["t2_tse_sag", "t2_tse_cor", "t2_tse_tra"]),
                                      [Column.ProxID,
                                       Column.DCMSerNum,
                                       Column.ijk])

for i, patient in enumerate(abnormal_list):
    abnormal_list[i][1] = str(abnormal_list[i][1]).split(" ")[2]
    
patients = []
PROSTATEx_path="/home/linas/Projects/Unsupervised-lesion-detection-with-multi-view-MRI-and-autoencoders/PROSTATEx"
for i, folder_path in enumerate(listdir(PROSTATEx_path)):
    patients.append(PatientDataPreprocessing(path.join(PROSTATEx_path, folder_path)))
    patients[i].extract(path.join("tets1", folder_path), abnormal_list)
