from pydicom import read_file

abnormal = read_file("test/ProstateX-0000/AXIAL/1-ProstateX-0000-4-9.dcm")
print(bool(abnormal[0x0051, 0x1014].value))
