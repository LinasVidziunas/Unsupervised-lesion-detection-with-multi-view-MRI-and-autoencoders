from Datapreprocessing.slice import Slice

from tensorflow import image, newaxis, constant
from numpy import array, concatenate

from os import listdir, path

class ProcessedData:
    """Import processed data with respect to the default folder structure"""

    def __init__(self, path_to_sets_folder: str = "../sets/"):
        if not path.isdir(path_to_sets_folder):
            raise ValueError(
                f"Path to the sets folder provided for ProcessedData class is not a valid directory: {path_to_sets_folder}"
            )

        self._path_to_sets_folder = path_to_sets_folder
        self._train = None
        self._validation = None
        self._test = None

    @property
    def train(self):
        if self._train is None:
            train_set_path = path.join(self._path_to_sets_folder + "train")
            self._train = Set(train_set_path)
        return self._train

    @property
    def validation(self):
        if self._validation is None:
            validation_set_path = path.join(self._path_to_sets_folder, "validation")
            self._validation = Set(validation_set_path)
        return self._validation

    @property
    def test(self):
        if self._test is None:
            test_set_path = path.join(self._path_to_sets_folder, "test")
            self._test = Set(test_set_path)
        return self._test

    def clear_data(self):
        self._train = None
        self._validation = None
        self._test = None


class Set:
    def __init__(self, path_to_set_folder: str):
        if not path.isdir(path_to_set_folder):
            raise ValueError(
                f"Path to the set folder provided for Set class is not a valid directory: {path_to_set_folder}"
            )

        self._path_to_set_folder = path_to_set_folder
        self._axial = None
        self._coronal = None
        self._sagittal = None

    @property
    def axial(self):
        if self._axial is None:
            axial_path = path.join(self._path_to_set_folder, "Axial")
            self._axial = View(axial_path)
        return self._axial

    @property
    def coronal(self):
        if self._coronal is None:
            coronal_path = path.join(self._path_to_set_folder, "Coronal")
            self._coronal = View(coronal_path)
        return self._coronal

    @property
    def sagittal(self):
        if self._sagittal is None:
            sagittal_path = path.join(self._path_to_set_folder, "Sagittal")
            self._sagittal = View(sagittal_path)
        return self._sagittal


class View:
    def __init__(self, path_to_view_in_set_folder: str, patient_id = None):
        if not path.isdir(path_to_view_in_set_folder):
            raise ValueError(
                f"Path to the view folder provided for the View class is not a valid directory: {path_to_view_in_set_folder}"
            )
        self._path_to_view_in_set = path_to_view_in_set_folder
        self._patient_id = patient_id
        self._slices = []

    @property
    def slices(self):
        if not self._slices:
            dicoms = listdir(self._path_to_view_in_set)

            # convert patiend_id: 1 -> "0001"
            if self._patient_id != None:
                for _, dicom in enumerate(dicoms):
                    if str(self._patient_id).zfill(4) in dicom:
                        _slice = Slice(path.join(self._path_to_view_in_set, dicom))
                        self._slices.append(_slice)
            else:
                for _, dicom in enumerate(dicoms):
                    _slice = Slice(path.join(self._path_to_view_in_set, dicom))
                    self._slices.append(_slice)

        return self._slices

    def get_abnormal_slices_as_normalized_pixel_arrays(self, shape=(320, 320)):
        """Returns abnormal resized slices as np.array object"""

        normalized_pixel_arrays = []

        for _, _slice in enumerate(self.slices):
            if _slice.get_abnormality() == 1:
                pixel_array = _slice.normalized_pixel_array()
                pixel_array = pixel_array[newaxis, ..., newaxis]
                pixel_array = image.resize(pixel_array, shape)[0, ..., 0].numpy()

                normalized_pixel_arrays.append(pixel_array)

        return array(normalized_pixel_arrays)

    def get_normal_slices_as_normalized_pixel_arrays(self, shape=(320, 320)):
        """Returns normal resized slices as np.array object"""

        normalized_pixel_arrays = []

        for _, _slice in enumerate(self.slices):
            if _slice.get_abnormality() == 0:
                pixel_array = _slice.normalized_pixel_array()
                pixel_array = pixel_array[newaxis, ..., newaxis]
                pixel_array = image.resize(pixel_array, shape)[0, ..., 0].numpy()
                
                normalized_pixel_arrays.append(pixel_array)

        return array(normalized_pixel_arrays)

    def get_slices_as_normalized_pixel_arrays(self, shape=(320, 320)):
        """Returns all slices as np.array object"""

        normalized_pixel_arrays = []

        for _, _slice in enumerate(self.slices):
            pixel_array = _slice.normalized_pixel_array()
            pixel_array = pixel_array[newaxis, ..., newaxis]
            pixel_array = image.resize(pixel_array, shape)[0, ..., 0].numpy()

            normalized_pixel_arrays.append(pixel_array)

        return array(normalized_pixel_arrays)


class Patient:
    def __init__(self, patient_id, path_to_set_folder: str = "../sets/train"):
        if not path.isdir(path_to_set_folder):
            raise ValueError(
                f"Path to the set folder provided for Patient class is not a valid directory: {path_to_set_folder}"
            )

        self.exists = [i for i in listdir(path.join(path_to_set_folder, "Axial")) if str(patient_id).zfill(4) in i] != []
        self.patient_id = patient_id
        self._path_to_set_folder = path_to_set_folder

        self._axial = None
        self._coronal = None
        self._sagittal = None

    @property
    def axial(self):
        if self._axial is None:
            axial_path = path.join(self._path_to_set_folder, "Axial")
            self._axial = View(axial_path, patient_id=self.patient_id)
        return self._axial

    @property
    def coronal(self):
        if self._coronal is None:
            coronal_path = path.join(self._path_to_set_folder, "Coronal")
            self._coronal = View(coronal_path, patient_id=self.patient_id)
        return self._coronal

    @property
    def sagittal(self):
        if self._sagittal is None:
            sagittal_path = path.join(self._path_to_set_folder, "Sagittal")
            self._sagittal = View(sagittal_path, patient_id=self.patient_id)
        return self._sagittal

    def get_normal_slices(self, shape: tuple, equal_amount: bool = False):
        axial = self.axial.get_normal_slices_as_normalized_pixel_arrays(shape)
        coronal = self.coronal.get_normal_slices_as_normalized_pixel_arrays(shape)
        sagittal = self.sagittal.get_normal_slices_as_normalized_pixel_arrays(shape)

        if equal_amount:
            minimum = min(len(axial), len(coronal), len(sagittal))
            axial = axial[:minimum-1]
            coronal = coronal[:minimum-1]
            sagittal = sagittal[:minimum-1]

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}

    def get_slices(self, shape: tuple, equal_amount: bool = False):
        axial = self.axial.get_slices_as_normalized_pixel_arrays(shape)
        coronal = self.coronal.get_slices_as_normalized_pixel_arrays(shape)
        sagittal = self.sagittal.get_slices_as_normalized_pixel_arrays(shape)

        if equal_amount:
            minimum = min(len(axial), len(coronal), len(sagittal))
            axial = axial[:minimum-1]
            coronal = coronal[:minimum-1]
            sagittal = sagittal[:minimum-1]

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}

    def clear_data(self):
        self._axial = None
        self._coronal = None
        self._sagittal = None


def get_data_by_patients(path_to_sets_folder: str = "../sets/", image_dim: tuple = (384, 384)):
    number_of_patients = 300 #ish

    train = {"axial": [], "coronal": [], "sagittal": []}

    val = {"axial": [], "coronal": [], "sagittal": []}
    y_val = {"axial": [], "coronal": [], "sagittal": []}

    test = {"axial": [], "coronal": [], "sagittal": []}
    y_test = {"axial": [], "coronal": [], "sagittal": []}

    for patient_id in range(number_of_patients+1):
        patient = Patient(patient_id, path.join(path_to_sets_folder, "train"))
        if not patient.exists:
            continue
        _slices = patient.get_normal_slices(shape=image_dim, equal_amount=True)
        train["axial"].append(_slices["axial"])
        train["coronal"].append(_slices["coronal"])
        train["sagittal"].append(_slices["sagittal"])

    val_patients = []
    for patient_id in range(number_of_patients+1):
        patient = Patient(patient_id, path.join(path_to_sets_folder, "validation"))
        if not patient.exists:
            continue
        val_patients.append(patient)

    for patient in val_patients:
        _slices = patient.get_slices(shape=image_dim, equal_amount=True)
        val["axial"].append(_slices["axial"])
        val["coronal"].append(_slices["coronal"])
        val["sagittal"].append(_slices["sagittal"])

    for patient in val_patients:
        for _slice in patient.axial.slices:
            y_val["axial"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in patient.coronal.slices:
            y_val["coronal"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in patient.sagittal.slices:
            y_val["sagittal"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])

        minimum = min(len(y_val["axial"]), len(y_val["coronal"]), len(y_val["sagittal"]))
        y_val["axial"] = y_val["axial"][:minimum-1]
        y_val["coronal"] = y_val["coronal"][:minimum-1]
        y_val["sagittal"] = y_val["sagittal"][:minimum-1]


    test_patients = []
    for patient_id in range(number_of_patients+1):
        patient = Patient(patient_id, path.join(path_to_sets_folder, "test"))
        if not patient.exists:
            continue
        test_patients.append(patient)

    for patient in test_patients:
        _slices = patient.get_slices(shape=image_dim, equal_amount=True)
        test["axial"].append(_slices["axial"])
        test["coronal"].append(_slices["coronal"])
        test["sagittal"].append(_slices["sagittal"])

    for patient in test_patients:
        for _slice in patient.axial.slices:
            y_test["axial"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in patient.coronal.slices:
            y_test["coronal"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in patient.sagittal.slices:
            y_test["sagittal"].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])

        minimum = min(len(y_test["axial"]), len(y_test["coronal"]), len(y_test["sagittal"]))
        y_test["axial"] = y_test["axial"][:minimum-1]
        y_test["coronal"] = y_test["coronal"][:minimum-1]
        y_test["sagittal"] = y_test["sagittal"][:minimum-1]

    train = {"axial": concatenate(train["axial"]), "coronal": concatenate(train["coronal"]), "sagittal": concatenate(train["sagittal"])}
    print(f"\n\n---------------------------- Start: Dataset information ----------------------------\n\n")
    print(f"Train dataset:")
    print(f"\tAxial:{len(train['axial'])}")
    print(f"\tCoronal:{len(train['coronal'])}")
    print(f"\tSagittal:{len(train['sagittal'])}")
    val = {"axial": concatenate(val["axial"]), "coronal": concatenate(val["coronal"]), "sagittal": concatenate(val["sagittal"])}
    print(f"Validation dataset:")
    print(f"\tAxial:{len(val['axial'])}")
    print(f"\tCoronal:{len(val['coronal'])}")
    print(f"\tSagittal:{len(val['sagittal'])}")
    test = {"axial": concatenate(test["axial"]), "coronal": concatenate(test["coronal"]), "sagittal": concatenate(test["sagittal"])}
    print(f"Test dataset:")
    print(f"\tAxial:{len(test['axial'])}")
    print(f"\tCoronal:{len(test['coronal'])}")
    print(f"\tSagittal:{len(test['sagittal'])}")

    return {
        "train": train,
        "validation": {"x": val, "y": y_val},
        "test": {"x": test, "y": y_test}
    }

def get_abnormality_tf_const(dataset):
    temp = ([[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in dataset.slices])
    return constant(temp, shape=(len(temp), 2))
