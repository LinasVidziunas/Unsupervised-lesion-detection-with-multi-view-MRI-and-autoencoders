import pydicom as dicom
import numpy as np
import os

def read_patient_image_orientation(dicom_file):
    Coronal_plane_view = [1, 0, 0, 0, 0, -1]
    Sagittal_plane_view = [0, 1, 0, 0, 0, -1]
    Axial_plane_view = [1, 0, 0, 0, 1, 0]
    orientation_vector = dicom_file.ImageOrientationPatient
    if orientation_vector == Coronal_plane_view:
        plane = "Coronal"
    if orientation_vector == Sagittal_plane_view:
        plane = "Sagittal"
    if orientation_vector == Axial_plane_view:
        plane = "Axial"
    return plane

def create_subfolders(patient_file_path):
    coronial_path = patient_file_path+"\\"+"Coronial"
    Sagittal_path = patient_file_path+"\\"+"Sagittal"
    Axial_path = patient_file_path+"\\"+"Axial"
    os.makedirs(Axial_path)
    os.makedirs(coronial_path)
    os.makedirs(Sagittal_path)


def get_paths_of_filenames(folder_path, list_of_filenames):
    folder_paths = []
    for filename in list_of_filenames:
        folder_paths.append(folder_path + '\\' + filename)
    return folder_paths

def from_DICOM_filepath_to_dicom(paths_Dicoms):
    list_of_DICOMs = []
    for path_DICOM in paths_DICOMs:
        list_of_DICOMs.append(dicom.read_file(path_DICOM))
    return list_of_DICOMs


patient_folder_counter=0
def build_patient_folder(to_path):
    global patient_folder_counter
    new_patient_folder_path = to_path + "\\" + str(patient_folder_counter)
    patient_folder_counter+=1
    os.makedirs(new_patient_folder_path)
    return new_patient_folder_path

#Her starter main koden

PROSTATEx_path="C:\\Users\\Orjan\\Desktop\\manifest-1642204867087\PROSTATEx"
folder_list = os.listdir(PROSTATEx_path)
folder_list.remove("LICENSE")
PROSTATEx_patient_folder_paths = get_paths_of_filenames(PROSTATEx_path, folder_list)

new_main_folder = "C:\\Users\\Orjan\\Desktop\\testing\\test3"

#hertil fungerer koden som den skal
for i in range(len(PROSTATEx_patient_folder_paths)):

    # For pasienten opprettes det en egen folder i new main folder, med subfolders(Axial,Coronal og Sagittal)
    new_patient_folder_path = build_patient_folder(new_main_folder)
    create_subfolders(new_patient_folder_path)

    current_folder = PROSTATEx_patient_folder_paths[i]
    folder_names_in_current_folder = os.listdir(current_folder)
    paths_of_subfolders = get_paths_of_filenames(current_folder, folder_names_in_current_folder)

    for j in range(len(paths_of_subfolders)):
        folder_with_DICOMs_names_list = os.listdir(paths_of_subfolders[j])
        for k, strings in enumerate(folder_with_DICOMs_names_list):
            if "t2tsesag" in strings:
                string = paths_of_subfolders[j] + "\\" + folder_with_DICOMs_names_list[k]
                filenames = os.listdir(string)
                paths_now = get_paths_of_filenames(string, filenames)
                for m, path in enumerate(paths_now):
                    a = dicom.read_file(path)
                    filename = str(a.SeriesInstanceUID)+"-"+str(a.InstanceNumber)+"-"+str(a.PatientID)+".csv"
                    np.savetxt(new_main_folder+"\\"+str(i)+"\\Sagittal"+"\\"+filename, a.pixel_array, delimiter=",")
            if "t2tsetra" in strings:
                string = paths_of_subfolders[j] + "\\" + folder_with_DICOMs_names_list[k]
                filenames = os.listdir(string)
                paths_now = get_paths_of_filenames(string ,filenames)
                for m, path in enumerate(paths_now):
                    a = dicom.read_file(path)
                    filename = str(a.SeriesInstanceUID)+"-"+str(a.InstanceNumber)+"-"+str(a.PatientID)+".csv"
                    np.savetxt(new_main_folder+"\\"+str(i)+"\\Axial"+"\\"+filename, a.pixel_array, delimiter=",")
            if "t2tsecor" in strings:
                string = paths_of_subfolders[j] + "\\" + folder_with_DICOMs_names_list[k]
                filenames = os.listdir(string)
                paths_now = get_paths_of_filenames(string ,filenames)
                for m, path in enumerate(paths_now):
                    a = dicom.read_file(path)
                    filename = str(a.SeriesInstanceUID)+"-"+str(a.InstanceNumber)+"-"+str(a.PatientID)+".csv"
                    np.savetxt(new_main_folder+"\\"+str(i)+"\\Coronial"+"\\"+filename, a.pixel_array, delimiter=",")
