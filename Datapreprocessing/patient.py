from Datapreprocessing.enums import View
from Datapreprocessing.slice import Slice

from os import path, makedirs, listdir


class Patient:

    def __init__(self, patient_folder_path: str, abnormal_list):
        self.__preprocessed_patient_folder_path = ""
        self.__processed_patient_folder_path = ""
        self.__slices = {}

        # Between patient folder path and the folder that contains DICOMs
        # there is a middle folder, skipping over this
        self.preprocessed_patient_folder_path = path.join(
            patient_folder_path,
            listdir(patient_folder_path)[0])

        # If folder name includes any of the oriantation protocol names,
        # add to dictionary of slices, where dict key is the orientation
        for folder_name in listdir(self.preprocessed_patient_folder_path):
            for view in View:
                if view.value in folder_name:
                    self.__slices[view.name] = self.__read_slices(
                        path.join(self.preprocessed_patient_folder_path,
                                  folder_name))

        # Foreach slice check whether it is in abnormal_list
        # and label it correspondigly
        for view, slices in self.__slices.items():
            for slice in self.__slices[view]:
                if [slice.patientID, slice.instanceNumber,
                        slice.seriesNumber] in abnormal_list:
                    slice.label_abnormality(True)
                else:
                    slice.label_abnormality(False)

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
        """Set path for processed patient data and
        And create them if they dont already exist.

        :param directory_path """

        if not path.exists(directory_path):
            try:
                makedirs(directory_path, exist_ok=True)
                for view in View:
                    makedirs(path.join(directory_path, view.name))
            except OSError:
                raise OSError(
                    f"Could not create a directory for processed patient information: {directory_path}"
                )

        self.__processed_patient_folder_path = directory_path

    def __read_slices(self, folder_path: str):
        """Returns a list of Slice instances of dicoms in folder_path"""
        return [
            Slice(path.join(folder_path, filename))
            for filename in listdir(folder_path)
        ]

    def extract(self, extract_to: str, remove_abnormal_slices: bool = False):
        """Saves slices"""
        self.processed_patient_folder_path = extract_to

        for view, slices in self.__slices.items():
            for slice in slices:
                if remove_abnormal_slices is True and slice.get_abnormality() == 1:
                    continue

                slice.save_as_dicom(
                    path.join(self.processed_patient_folder_path, view))
