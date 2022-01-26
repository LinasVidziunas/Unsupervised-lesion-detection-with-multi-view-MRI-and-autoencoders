from pydicom import read_file
from patient_data_preprocessing import Slice

abnormal = read_file("test/ProstateX-0000/Axial/1-ProstateX-0000-4-9.dcm")
# print(bool(abnormal[0x0051, 0x1014].value))

ab = Slice("test/ProstateX-0000/Axial/1-ProstateX-0000-4-9.dcm")
no = Slice("test/ProstateX-0001/Coronal/0-ProstateX-0001-5-1.dcm")

print(ab.dicom.pixel_array)
print(ab.normalized_pixel_array())
