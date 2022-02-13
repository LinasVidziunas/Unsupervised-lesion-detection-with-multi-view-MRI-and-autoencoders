from Datapreprocessing.slice import Slice

from tensorflow import image, newaxis
from numpy import array

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
            train_set_path = self._path_to_sets_folder + "train/"
            self._train = Set(train_set_path)
        return self._train

    @property
    def validation(self):
        if self._validation is None:
            validation_set_path = self._path_to_sets_folder + "validation/"
            self._validation = Set(validation_set_path)
        return self._validation

    @property
    def test(self):
        if self._test is None:
            test_set_path = self._path_to_sets_folder + "test/"
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
            axial_path = self._path_to_set_folder + "Axial/"
            self._axial = View(axial_path)
        return self._axial

    @property
    def coronal(self):
        if self._coronal is None:
            coronal_path = self._path_to_set_folder + "Coronal/"
            self._coronal = View(coronal_path)
        return self._coronal

    @property
    def sagittal(self):
        if self._sagittal is None:
            sagittal_path = self._path_to_set_folder + "Sagittal/"
            self._sagittal = View(sagittal_path)
        return self._sagittal


class View:
    def __init__(self, path_to_view_in_set_folder: str):
        if not path.isdir(path_to_view_in_set_folder):
            raise ValueError(
                f"Path to the view folder provided for the View class is not a valid directory: {path_to_view_in_set_folder}"
            )
        self._path_to_view_in_set = path_to_view_in_set_folder
        self._slices = []

    @property
    def slices(self):
        if not self._slices:
            dicoms = listdir(self._path_to_view_in_set)

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
