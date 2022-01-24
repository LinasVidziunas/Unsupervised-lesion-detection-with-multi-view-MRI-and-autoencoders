from csv import reader
from PatientDataExtractor.enums import Column


class PatientDataExtractor:
    __data = []

    @property
    def data(self):
        """All the data read when instance is initialized."""
        return self.__data

    def __init__(self, file_location: str = "ProstateX-Images-Train.csv"):
        with open(file_location, 'r', newline='') as csvfile:
            for row_num, row in enumerate(reader(csvfile, delimiter=",")):
                # Continue on first row (the header)
                if (row_num == 0):
                    continue

                if (len(row) < 1):
                    continue

                self.__data.append(row)

    def filter_by_column(self, column: Column, keywords: [str] = [""]):
        """Filter rows by keyword in a specific column."""
        rows = []

        for patient_data in self.__data:
            for keyword in keywords:
                if keyword in patient_data[
                        column.value] and patient_data not in rows:
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
