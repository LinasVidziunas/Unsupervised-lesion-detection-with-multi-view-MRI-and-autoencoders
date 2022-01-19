from dataextractor import PatientDataExtractor, keep_columns, Column
from patient_data_preprocessing import get_paths_of_filenames
from os import listdir, path, rename





ProstateX_Images_Train = "C:\\Users\\Orjan\\Desktop\\Bakkelor2022\\cancer_excel\\ProstateX-2-Images-Train.csv"

new_main_folder = "C:\\Users\\Orjan\\Desktop\\testing\\test3"
list_of_patient_folders = listdir(new_main_folder)
paths_of_patient_folder = get_paths_of_filenames(new_main_folder, list_of_patient_folders)


"""Get data"""
data = PatientDataExtractor(ProstateX_Images_Train)
sagittal_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_sag"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])
coronal_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_cor"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])
axial_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_tra"),
                             [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])

cancer = []
not_cancer = []

def check_for_cancer(orientation_folder, orientation_data):
    dicom_file_name = listdir(orientation_folder)
    for string in dicom_file_name:
        list_of_info = string.split("-")
        current_filename_path = path.join(orientation_folder, string)
        for arrays in orientation_data:
            if list_of_info[0] == arrays[3] and list_of_info[1] == arrays[1].split(" ")[2]:
                cancer.append(current_filename_path)
            else:
                not_cancer.append(current_filename_path)

def label_files(list_cancer, list_not_cancer):
    for files in list_cancer:
        files_split = path.split(files)
        omg_sa_mye_kode = "1-" + files_split[-1]
        rename(path.join(files_split[0], files_split[1]), path.join(files_split[0], omg_sa_mye_kode))
    for file in list_not_cancer:
        file_split = path.split(file)
        sa_mye_kode = "0-" + file_split[-1]
        rename(path.join(file_split[0], file_split[1]), path.join(file_split[0], sa_mye_kode))

# directory = '\some\folder\elsewhere'
# os.rename(os.path.join(directory, 'filename.txt'), os.path.join(directory, 'filename2.txt'))



for patient_file_path in paths_of_patient_folder:
    list_of_orientation_folders = listdir(patient_file_path)
    for string_name in list_of_orientation_folders:
        if string_name == "SAGITTAL":
            folder_path = path.join(patient_file_path, "SAGITTAL")
            check_for_cancer(folder_path, sagittal_data)
        if string_name == "CORONAL":
            folder_path = path.join(patient_file_path, "CORONAL")
            check_for_cancer(folder_path, coronal_data)
        if string_name == "AXIAL":
            folder_path = path.join(patient_file_path, "AXIAL")
            check_for_cancer(folder_path, axial_data)


# test_list =["a","a","b","b"]
# mylist = list(dict.fromkeys(test_list))
# print(mylist)
list_cancer = list(dict.fromkeys(cancer))
list_not_cancer = list(dict.fromkeys(not_cancer))
for items in list_cancer:
    if items in list_not_cancer:
        list_not_cancer.remove(items)
for tokens in list_not_cancer:
    if tokens in list_cancer:
        list_cancer.remove(tokens)

label_files(list_cancer, list_not_cancer)

