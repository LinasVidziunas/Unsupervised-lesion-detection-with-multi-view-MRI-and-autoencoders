from pydicom import read_file

from os import path


class Slice:

    def __init__(self, path):
        self.dicom = read_file(path)
        self.patientID = self.dicom.PatientID
        self.seriesNumber = self.dicom.SeriesNumber
        self.instanceNumber = self.dicom.InstanceNumber
        self.protocol = self.dicom.ProtocolName

    def label_abnormality(self, abnormal: bool):
        self.abnormal = abnormal
        self.dicom.add_new([0x0051, 0x1014], 'US', abnormal)

    def get_abnormality(self):
        if [0x0051, 0x1014] in self.dicom:
            self.abnormal = self.dicom[0x0051, 0x1014].value
            return int(self.abnormal)
        return None

    def save_as_dicom(self, directory_path):
        if self.get_abnormality() is not None:
            filename = f"{int(self.abnormal)}-{self.patientID}-{self.seriesNumber}-{self.instanceNumber}.dcm"
            self.dicom.save_as(path.join(directory_path, filename),
                               write_like_original=False)

    def normalized_pixel_array(self):
        pixel_array = self.dicom.pixel_array
        return pixel_array / self.dicom.LargestImagePixelValue
