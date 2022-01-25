from enum import Enum


class View(Enum):
    Sagittal = "t2tsesag"
    Coronal = "t2tsecor"
    Axial = "t2tsetra"


class Column(Enum):
    ProxID = 0  # Patient Id
    Name = 1
    fid = 2  # studydate = 2
    pos = 3
    WorldMatrix = 4
    ijk = 5
    toplevel = 6
    SpacingBetweenSlices = 7
    VoxelSpacing = 8
    Dim = 9
    DCMSerDescr = 10  # Some type of description
    DCMSerNum = 11
