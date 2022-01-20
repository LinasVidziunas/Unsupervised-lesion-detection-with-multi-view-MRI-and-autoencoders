from os import path, makedirs, listdir, path, makedirs
from enum import Enum

from pydicom import read_file
from numpy import savetxt


class View(Enum):
    SAGITTAL = [0, 1, 0, 0, 0, -1]
    CORONAL = [1, 0, 0, 0, 0, -1]
    AXIAL = [1, 0, 0, 0, 1, 0]


class PatientDataPreprocessing:
    __preprocessed_patient_folder_path = ""
    __processed_patient_folder_path = ""

    @property
    def preprocessed_patient_folder_path(self):
        return self.__preprocessed_patient_folder_path

    @preprocessed_patient_folder_path.setter
    def preprocessed_patient_folder_path(self, directory_path):
        if not path.exists(directory_path):
            raise OSError(f"Trying to set an invalid patient folder path: {directory_path}")

        self.__preprocessed_patient_folder_path = directory_path

    @property
    def processed_patient_folder_path(self):
        return self.__processed_patient_folder_path

    @processed_patient_folder_path.setter
    def processed_patient_folder_path(self, directory_path):
        if not path.exists(directory_path):
            try:
                makedirs(directory_path, exist_ok=True)
            except OSError:
                raise OSError(f"Could not create a directory for processed patient information: {directory_path}")

        self.__processed_patient_folder_path = directory_path

    def __init__(self, patient_folder_path: str):
        self.preprocessed_patient_folder_path = path.join(patient_folder_path, listdir(patient_folder_path)[0])

    def __create_subfolders(self):
        for view in View:
            makedirs(path.join(self.processed_patient_folder_path, view.name))

    def read_patient_image_orientation(self, dicom_file):
        orientation_vector = dicom_file.ImageOrientationPatient
        if orientation_vector == View.CORONAL.value:
            return "Coronal"
        if orientation_vector == View.SAGITTAL.value:
            return "Sagittal"
        if orientation_vector == View.AXIAL.value:
            return "Axial"

    def __read_DICOMs(self, folder_path: str):
        DICOMS = []

        filenames = listdir(folder_path)

        for filename in filenames:
            DICOMS.append(read_file(path.join(folder_path, filename)))

        return DICOMS

    def __save_DICOM_as_txt(self, DICOM, folder_path):
        filename = str(DICOM.SeriesInstanceUID) + "-" + str(DICOM.InstanceNumber) + "-" + str(DICOM.PatientID) + ".csv"
        savetxt(path.join(folder_path, filename), DICOM.pixel_array, delimiter=',')

    def extract(self, extract_to: str):
        self.processed_patient_folder_path = extract_to
        self.__create_subfolders()
        for folder_name in listdir(self.preprocessed_patient_folder_path):
            if "t2tsesag" in folder_name:
                DICOMS = self.__read_DICOMs(path.join(self.preprocessed_patient_folder_path, folder_name))

                for DICOM in DICOMS:
                    self.__save_DICOM_as_txt(DICOM, path.join(self.processed_patient_folder_path, "SAGITTAL"))
                    
            if "t2tsetra" in folder_name:
                DICOMS = self.__read_DICOMs(path.join(self.preprocessed_patient_folder_path, folder_name))
                
                for DICOM in DICOMS:
                    self.__save_DICOM_as_txt(DICOM, path.join(self.processed_patient_folder_path, "AXIAL"))
                   
            if "t2tsecor" in folder_name:
                DICOMS = self.__read_DICOMs(path.join(self.preprocessed_patient_folder_path, folder_name))
                
                for DICOM in DICOMS:
                    self.__save_DICOM_as_txt(DICOM, path.join(self.processed_patient_folder_path, "CORONAL"))

def get_paths_of_filenames(folder_path, list_of_filenames):
    folder_paths = []
    for filename in list_of_filenames:
        folder_paths.append(path.join(folder_path, filename))
    return folder_paths


if __name__ == "__main__":
    patients = []

    PROSTATEx_path="C:\\Users\\Orjan\\Desktop\\manifest-1642204867087\PROSTATEx"
    patient_folder_list = listdir(PROSTATEx_path)
    patient_folder_list.remove("LICENSE")
    PROSTATEx_patient_folder_paths = get_paths_of_filenames(PROSTATEx_path, patient_folder_list)
    
    new_main_folder = "C:\\Users\\Orjan\\Desktop\\Bakkelor2022\\testing\\test3"
    
    # for i, folder_path in enumerate(PROSTATEx_patient_folder_paths):
    #     patients.append(PatientDataPreprocessing(folder_path))
    #     patients[i].extract(path.join(new_main_folder, str(i)))
#testfile
testpath ="F:\\Bakkelor2022\\manifest-A3Y4AE4o5818678569166032044\\PROSTATEx\\ProstateX-0008\\10-21-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-64134\\6.000000-t2tsesag-69918\\1-01.dcm"
testpath2 ="F:\\Bakkelor2022\\manifest-A3Y4AE4o5818678569166032044\\PROSTATEx\\ProstateX-0008\\10-21-2011-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-64134\\3.000000-t2tsesag-46088\\1-01.dcm"
dicom_test = read_file(testpath)
dicom_test2 =read_file(testpath2)
print(dicom_test2.SeriesNumber)
