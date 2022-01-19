from enum import Enum


class View(Enum):
    SAGITTAL = [0, 1, 0, 0, 0, -1]
    CORONAL = [1, 0, 0, 0, 0, -1]
    AXIAL = [1, 0, 0, 0, 1, 0]


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
