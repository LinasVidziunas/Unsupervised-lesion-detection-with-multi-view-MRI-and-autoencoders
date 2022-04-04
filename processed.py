from Datapreprocessing.slice import Slice

from tensorflow import image, newaxis, constant
from numpy import array, concatenate

from os import listdir, path
from math import ceil

def equal_length_views_middle(x, y , z):
    minimum = min(len(x), len(y), len(z))

    # Sick code. Retrieves slices in the middle with length equal to minimum.
    x = x[(len(x)//2)-(minimum//2):len(x)-(ceil(len(x)/2)-ceil(minimum/2))]
    y = y[(len(y)//2)-(minimum//2):len(y)-(ceil(len(y)/2)-ceil(minimum/2))]
    z = z[(len(z)//2)-(minimum//2):len(z)-(ceil(len(z)/2)-ceil(minimum/2))]

    return x, y, z


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
    def __init__(self, path_to_view_in_set_folder: str, patient_id = None, sort_slices:bool = False):
        if not path.isdir(path_to_view_in_set_folder):
            raise ValueError(
                f"Path to the view folder provided for the View class is not a valid directory: {path_to_view_in_set_folder}"
            )
        self._path_to_view_in_set = path_to_view_in_set_folder
        self._patient_id = patient_id
        self._slices = []
        self._sort_slices = sort_slices

    @property
    def slices(self):
        if not self._slices:
            dicoms = listdir(self._path_to_view_in_set)

            if self._sort_slices:
                dicoms.sort(key=lambda el: el[-2:])

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
            self._axial = View(axial_path, patient_id=self.patient_id, sort_slices=True)
        return self._axial

    @property
    def coronal(self):
        if self._coronal is None:
            coronal_path = path.join(self._path_to_set_folder, "Coronal")
            self._coronal = View(coronal_path, patient_id=self.patient_id, sort_slices=True)
        return self._coronal

    @property
    def sagittal(self):
        if self._sagittal is None:
            sagittal_path = path.join(self._path_to_set_folder, "Sagittal")
            self._sagittal = View(sagittal_path, patient_id=self.patient_id, sort_slices=True)
        return self._sagittal

    def get_normal_slices(self, shape: tuple, equal_amount: bool = False):
        axial = self.axial.get_normal_slices_as_normalized_pixel_arrays(shape)
        coronal = self.coronal.get_normal_slices_as_normalized_pixel_arrays(shape)
        sagittal = self.sagittal.get_normal_slices_as_normalized_pixel_arrays(shape)

        if equal_amount:
            axial, coronal, sagittal = equal_length_views_middle(axial, coronal, sagittal)

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}

    def get_slices(self, shape: tuple, equal_amount: bool = False):
        axial = self.axial.get_slices_as_normalized_pixel_arrays(shape)
        coronal = self.coronal.get_slices_as_normalized_pixel_arrays(shape)
        sagittal = self.sagittal.get_slices_as_normalized_pixel_arrays(shape)

        if equal_amount:
            axial, coronal, sagittal = equal_length_views_middle(axial, coronal, sagittal)

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}

    def clear_data(self):
        self._axial = None
        self._coronal = None
        self._sagittal = None


def get_data_by_patients(path_to_sets_folder: str = "../sets/", image_dim: tuple = (384, 384)):
    number_of_patients = 300 #ish

    train = {"axial": [], "coronal": [], "sagittal": []}

    val = {"axial": [], "coronal": [], "sagittal": []}
    val_axial_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    val_coronal_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    val_sagittal_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    val_axial_normal = {"axial": [], "coronal": [], "sagittal": []}
    val_coronal_normal = {"axial": [], "coronal": [], "sagittal": []}
    val_sagittal_normal = {"axial": [], "coronal": [], "sagittal": []}
    y_val = {"axial": [], "coronal": [], "sagittal": []}

    test = {"axial": [], "coronal": [], "sagittal": []}
    test_axial_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    test_coronal_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    test_sagittal_abnormal = {"axial": [], "coronal": [], "sagittal": []}
    test_axial_normal = {"axial": [], "coronal": [], "sagittal": []}
    test_coronal_normal = {"axial": [], "coronal": [], "sagittal": []}
    test_sagittal_normal = {"axial": [], "coronal": [], "sagittal": []}
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
        axial, coronal, sagittal = equal_length_views_middle(patient.axial.slices, patient.coronal.slices, patient.sagittal.slices)

        for _slice in axial:
            y_val['axial'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in coronal:
            y_val['coronal'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in sagittal:
            y_val['sagittal'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])

        for i, _slice in enumerate(axial):
            if _slice.get_abnormality():
                val_axial_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_axial_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_axial_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                val_axial_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_axial_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_axial_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))

        for i, _slice in enumerate(coronal):
            if _slice.get_abnormality():
                val_coronal_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_coronal_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_coronal_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                val_coronal_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_coronal_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_coronal_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))

        for i, _slice in enumerate(sagittal):
            if _slice.get_abnormality():
                val_sagittal_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_sagittal_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_sagittal_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                val_sagittal_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                val_sagittal_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                val_sagittal_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))

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
        axial, coronal, sagittal = equal_length_views_middle(patient.axial.slices, patient.coronal.slices, patient.sagittal.slices)

        for _slice in axial:
            y_test['axial'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in coronal:
            y_test['coronal'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])
        for _slice in sagittal:
            y_test['sagittal'].append([int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()])

        for i, _slice in enumerate(axial):
            if _slice.get_abnormality():
                test_axial_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_axial_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_axial_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                test_axial_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_axial_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_axial_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))

        for i, _slice in enumerate(coronal):
            if _slice.get_abnormality():
                test_coronal_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_coronal_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_coronal_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                test_coronal_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_coronal_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_coronal_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))

        for i, _slice in enumerate(sagittal):
            if _slice.get_abnormality():
                test_sagittal_abnormal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_sagittal_abnormal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_sagittal_abnormal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))
            else:
                test_sagittal_normal["axial"].append(axial[i].normalized_reshaped_pixel_array(image_dim))
                test_sagittal_normal["coronal"].append(coronal[i].normalized_reshaped_pixel_array(image_dim))
                test_sagittal_normal["sagittal"].append(sagittal[i].normalized_reshaped_pixel_array(image_dim))


    train = {"axial": concatenate(train["axial"]), "coronal": concatenate(train["coronal"]), "sagittal": concatenate(train["sagittal"])}
    print(f"\n\n---------------------------- Start: Dataset information ----------------------------\n\n")
    print(f"Train dataset:")
    print(f"\tAxial:{len(train['axial'])}")
    print(f"\tCoronal:{len(train['coronal'])}")
    print(f"\tSagittal:{len(train['sagittal'])}")
    val = {"axial": concatenate(val["axial"]), "coronal": concatenate(val["coronal"]), "sagittal": concatenate(val["sagittal"])}
    print(f"Validation dataset:")
    print(f"\tAxial:{len(val['axial'])} where {len(val_axial_normal['axial'])} are normal and {len(val_axial_abnormal['axial'])} abnormal")
    print(f"\tCoronal:{len(val['coronal'])} where {len(val_coronal_normal['coronal'])} are normal and {len(val_coronal_abnormal['coronal'])} abnormal")
    print(f"\tSagittal:{len(val['sagittal'])} where {len(val_sagittal_normal['sagittal'])} are normal and {len(val_sagittal_abnormal['sagittal'])} abnormal")
    test = {"axial": concatenate(test["axial"]), "coronal": concatenate(test["coronal"]), "sagittal": concatenate(test["sagittal"])}
    print(f"Test dataset:")
    print(f"\tAxial:{len(test['axial'])} where {len(test_axial_normal['axial'])} are normal and {len(test_axial_abnormal['axial'])} abnormal")
    print(f"\tCoronal:{len(test['coronal'])} where {len(test_coronal_normal['coronal'])} are normal and {len(test_coronal_abnormal['coronal'])} abnormal")
    print(f"\tSagittal:{len(test['sagittal'])} where {len(test_sagittal_normal['sagittal'])} are normal and {len(test_sagittal_abnormal['sagittal'])} abnormal")

    val_axial_normal = {"axial": array(val_axial_normal["axial"]), "coronal": array(val_axial_normal["coronal"]), "sagittal": array(val_axial_normal["sagittal"])}
    val_coronal_normal = {"axial": array(val_coronal_normal["axial"]), "coronal": array(val_coronal_normal["coronal"]), "sagittal": array(val_coronal_normal["sagittal"])}
    val_sagittal_normal = {"axial": array(val_sagittal_normal["axial"]), "coronal": array(val_sagittal_normal["coronal"]), "sagittal": array(val_sagittal_normal["sagittal"])}
    val_axial_abnormal = {"axial": array(val_axial_abnormal["axial"]), "coronal": array(val_axial_abnormal["coronal"]), "sagittal": array(val_axial_abnormal["sagittal"])}
    val_coronal_abnormal = {"axial": array(val_coronal_abnormal["axial"]), "coronal": array(val_coronal_abnormal["coronal"]), "sagittal": array(val_coronal_abnormal["sagittal"])}
    val_sagittal_abnormal = {"axial": array(val_sagittal_abnormal["axial"]), "coronal": array(val_sagittal_abnormal["coronal"]), "sagittal": array(val_sagittal_abnormal["sagittal"])}
    test_axial_normal = {"axial": array(test_axial_normal["axial"]), "coronal": array(test_axial_normal["coronal"]), "sagittal": array(test_axial_normal["sagittal"])}
    test_coronal_normal = {"axial": array(test_coronal_normal["axial"]), "coronal": array(test_coronal_normal["coronal"]), "sagittal": array(test_coronal_normal["sagittal"])}
    test_sagittal_normal = {"axial": array(test_sagittal_normal["axial"]), "coronal": array(test_sagittal_normal["coronal"]), "sagittal": array(test_sagittal_normal["sagittal"])}
    test_axial_abnormal = {"axial": array(test_axial_abnormal["axial"]), "coronal": array(test_axial_abnormal["coronal"]), "sagittal": array(test_axial_abnormal["sagittal"])}
    test_coronal_abnormal = {"axial": array(test_coronal_abnormal["axial"]), "coronal": array(test_coronal_abnormal["coronal"]), "sagittal": array(test_coronal_abnormal["sagittal"])}
    test_sagittal_abnormal = {"axial": array(test_sagittal_abnormal["axial"]), "coronal": array(test_sagittal_abnormal["coronal"]), "sagittal": array(test_sagittal_abnormal["sagittal"])}


    # convert numpy arrays to tf.constants
    # y_val = {
    #     "axial": constant(y_val['axial'], shape=(len(y_val['axial']), 2)),
    #     "coronal": constant(y_val['coronal'], shape=(len(y_val['coronal']), 2)),
    #     "sagittal": constant(y_val['sagittal'], shape=(len(y_val['sagittal']), 2))
    # }

    # y_test = {
    #     "axial": constant(y_test['axial'], shape=(len(y_test['axial']), 2)),
    #     "coronal": constant(y_test['coronal'], shape=(len(y_test['coronal']), 2)),
    #     "sagittal": constant(y_test['sagittal'], shape=(len(y_test['sagittal']), 2))
    # }

    y_val = {"axial": array(y_val['axial']), "coronal": array(y_val['coronal']), "sagittal": array(y_val['sagittal'])}
    y_test = {"axial": array(y_test['axial']), "coronal": array(y_test['coronal']), "sagittal": array(y_test['sagittal'])}

    return {
        "train": train,
        "validation": {"x": val, "y": y_val},
        "val_normal": {"axial": val_axial_normal, "coronal": val_coronal_normal, "sagittal": val_sagittal_normal},
        "val_abnormal": {"axial": val_axial_abnormal, "coronal": val_coronal_abnormal, "sagittal": val_sagittal_abnormal},
        "test": {"x": test, "y": y_test},
        "test_normal": {"axial": test_axial_normal, "coronal": test_coronal_normal, "sagittal": test_sagittal_normal},
        "test_abnormal": {"axial": test_axial_abnormal, "coronal": test_coronal_abnormal, "sagittal": test_sagittal_abnormal},
    }

def get_abnormality_tf_const(dataset):
    temp = ([[int(not (bool(_slice.get_abnormality()))), _slice.get_abnormality()] for _slice in dataset.slices])
    return constant(temp, shape=(len(temp), 2))
