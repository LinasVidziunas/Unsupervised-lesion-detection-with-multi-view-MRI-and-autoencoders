from dataextractor import PatientDataExtractor, keep_columns, Column
from folder_processing import get_paths_of_filenames
from os import listdir, path


ProstateX-2-Images-Train = ""

new_main_folder = ""
list_of_patient_folders = listdir(new_main_folder)
paths_of_patient_folder = get_paths_of_filenames(new_main_folder,list_of_patient_folders)


"""Get data"""
data = PatientDataExtractor(ProstateX-2-Images-Train)
sagittal_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_sag"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])
coronal_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_cor"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])
axial_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_tra"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])

1.3.6.1.4.1.14519.5.2.1.7311.5101.180041601751316950966041961765-1-ProstateX-0009

def label_filenames(orientation_folder,orientation_data):
    for k, string in enumerate(orientation_folder):
        list_of_info = string.split("-")
        if list_of_info[0] in sagittal_data[3]:
            


for patient_file_path in paths_of_patient_folder:
    list_of_orientation_folders = listdir(new_main_folder)
    for string_name in list_of_orientation_folders:
        if string_name == "SAGITTAL":
            folder_path = path.join(patient_file_path,"SAGITTAL")
            label_filenames(folder_path,sagittal_data)
        if string_name == "CORONAL":
            folder_path = path.join(patient_file_path, "CORONAL")
            label_filenames(folder_path,coronal_data)
        if string_name == "AXIAL":
            folder_path = path.join(patient_file_path, "AXIAL")
            label_filenames(folder_path,axial_data)

