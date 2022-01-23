from os import path, makedirs, listdir

from pydicom import read_file
from PatientDataExtractor.enums import View


class PatientDataPreprocessing:
    __preprocessed_patient_folder_path = ""
    __processed_patient_folder_path = ""

    def __init__(self, patient_folder_path: str):
        # Between patient folder path and the folder that contains DICOMs there is a middle folder, skipping over this:
        self.preprocessed_patient_folder_path = path.join(
            patient_folder_path,
            listdir(patient_folder_path)[0])

    @property
    def preprocessed_patient_folder_path(self):
        return self.__preprocessed_patient_folder_path

    @preprocessed_patient_folder_path.setter
    def preprocessed_patient_folder_path(self, directory_path):
        if not path.exists(directory_path):
            raise OSError(
                f"Trying to set an invalid patient folder path: {directory_path}"
            )

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
                raise OSError(
                    f"Could not create a directory for processed patient information: {directory_path}"
                )

        self.__processed_patient_folder_path = directory_path

    def __create_sub_folders(self):
        for view in View:
            makedirs(path.join(self.processed_patient_folder_path, view.name))

    def __read_DICOMs(self, folder_path: str):
        DICOMS = []

        filenames = listdir(folder_path)

        for filename in filenames:
            DICOMS.append(read_file(path.join(folder_path, filename)))

        return DICOMS

    def __save_DICOM_as_DICOM(self, DICOM, folder_path, cancer_list):
        abnormal_variable = 0

        if [DICOM.PatientID, DICOM.InstanceNumber,
                DICOM.SeriesNumber] in cancer_list:
            abnormal_variable = 1

        filename = str(abnormal_variable) + "-" + str(
            DICOM.PatientID) + "-" + str(DICOM.SeriesNumber) + "-" + str(
                DICOM.InstanceNumber) + ".dcm"
        DICOM.add_new([0x0051, 0x1014], 'US', abnormal_variable)
        DICOM.save_as(path.join(folder_path, filename),
                      write_like_original=False)

    def extract(self, extract_to: str, list_of_patients_abnormal):
        self.processed_patient_folder_path = extract_to
        self.__create_sub_folders()

        orientation = {
            "t2tsesag": "SAGITTAL",
            "t2tsetra": "AXIAL",
            "t2tsecor": "CORONAL"
        }

        for folder_name in listdir(self.preprocessed_patient_folder_path):
            for key, value in orientation.items():
                if key in folder_name:
                    DICOMS = self.__read_DICOMs(
                        path.join(self.preprocessed_patient_folder_path,
                                  folder_name))

                    for DICOM in DICOMS:
                        self.__save_DICOM_as_DICOM(
                            DICOM,
                            path.join(self.processed_patient_folder_path,
                                      value), list_of_patients_abnormal)


def get_paths_of_filenames(folder_path, list_of_filenames):
    folder_paths = []
    for filename in list_of_filenames:
        folder_paths.append(path.join(folder_path, filename))
    return folder_paths
