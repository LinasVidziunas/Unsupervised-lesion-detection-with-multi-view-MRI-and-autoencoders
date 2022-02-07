from Datapreprocessing.slice import Slice

from os import listdir, path


class ProcessedData:
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
            train_set_path = self._path_to_sets_folder + "Train/"
            self._train = Set(train_set_path)
        return self._train

    @property
    def validation(self):
        if self._train is None:
            validation_set_path = self._path_to_sets_folder + "Validation/"
            self._validation = Set(validation_set_path)
        return self._validation

    @property
    def test(self):
        if self._train is None:
            test_set_path = self._path_to_sets_folder + "Test/"
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

    @property()
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
        self._slices = None

    @property
    def slices(self):
        if self._slices is None:
            dicoms = listdir(self._path_to_view_in_set)

            for i, dicom in enumerate(dicoms):
                _slice = Slice(path.join(self._path_to_view_in_set, dicom))
                self._slices.append(_slice)

        return self._slices

    def get_slices_as_normalized_pixel_arrays(self, shape=(320, 320)):
        normalized_pixel_arrays = []

        for i, _slice in enumerate(self.slices()):
            normalized_pixel_array = _slice.normalized_pixel_array()
            if normalized_pixel_array.shape == shape:
                normalized_pixel_arrays.append(normalized_pixel_array)

        return normalized_pixel_arrays

# from os import listdir, path

# train_files = listdir("sets/x_train")
# test_files = listdir("sets/x_test")

# x_train = np.zeros((len(train_files), 320, 320))
# x_test = np.zeros((len(test_files), 320, 320))

# train_slices = []
# test_slices = []

# # ValueError thrown when slice does not match the default resolution
# for i, slice_file in enumerate(train_files):
#     try:
#         _slice = Slice(path.join("sets/x_train", slice_file))
#         x_train[i][:][:] = _slice.normalized_pixel_array()
#         train_slices.append(_slice)
#     except ValueError:
#         x_train[i][:][:] = x_train[i - 1][:][:]

# for i, slice_file in enumerate(test_files):
#     try:
#         _slice = Slice(path.join("sets/x_test", slice_file))
#         x_test[i][:][:] = _slice.normalized_pixel_array()
#         test_slices.append(_slice)
#     except ValueError:
#         x_test[i][:][:] = x_test[i - 1][:][:]
