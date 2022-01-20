from csv import reader
from enum import Enum


class View(Enum):
    SAGGITAL = 0
    CORONAL = 1
    AXIAL = 2


class Column(Enum):
    ProxID = 0  # Patient Id
    Name = 1
    fid = 2 #studydate = 2
    pos = 3
    WorldMatrix = 4
    ijk = 5
    toplevel = 6
    SpacingBetweenSlices = 7
    VoxelSpacing = 8
    Dim = 9
    DCMSerDescr = 10  # Some type of description
    DCMSerNum = 11


class PatientDataExtractor:
    __data = []

    @property
    def data(self):
        """All the data read when instance is initialized."""
        return __data


    def __init__(self, file_location: str ="prostate_train.csv"):
        with open(file_location, 'r', newline='') as csvfile:
            for row_num, row in enumerate(reader(csvfile, delimiter=",")):

                # Continue on first row (the header)
                if (row_num == 0):
                    continue

                if (len(row) < 1):
                    continue

                self.__data.append(row)

    def filter_by_column(self, column: Column, keywords: [str] = [""]):
        """Filter rows by keyword in a specific column. BIG FUNCTION MISTAKE. DO NOT USE"""
        rows = []

        for patient_data in self.__data:
            for keyword in keywords:
                if keyword in patient_data[column.value]:
                    rows.append(patient_data)
        return rows

def keep_columns(data, columns: [Column]):
    return_data = []
    indexes = [column.value for column in columns]

    for patient_data in data:
        patient = []

        for column_index, column in enumerate(patient_data):
            if column_index in indexes:
                patient.append(column)
        
        return_data.append(patient)

    return return_data


if __name__ == "__main__":
    data = PatientDataExtractor("C:\\Users\\Orjan\\Desktop\\Bakkelor2022\\cancer_excel\\prostate_train.csv")
    filtered_data = keep_columns(data.filter_by_column(Column.DCMSerDescr, "t2_tse_sag"),
                                 [Column.ProxID, Column.ijk, Column.DCMSerDescr, Column.DCMSerUID])
    print(filtered_data)
