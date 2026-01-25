from enum import Enum, auto


class GemmDimension(Enum):
    ONE = auto()
    PX = auto()
    PY = auto()
    SIZE = auto()

class MatrixCommunicated(Enum):
    A = auto()
    B = auto()
    C = auto()

class SubtileScheme(Enum):
    COL = auto() # split the matrix along the columns
    ROW = auto() # split the matrix along the rows

class CommunicationDirection(Enum):
    SEND_NEXT = auto() # send subtiles to a higher rank and receive from a lower one
    SEND_PREV = auto()

class ConfigurationOptions1D(Enum):
    DIVISIBILITY = auto()
    DISTRIBUTION = auto()
    GET_LOCAL_INDICES = auto()
    INDEX = auto()
    BUFFER = auto()
    CURRENT_TILES = auto()
    SET_C = auto()
    DIRECTION_INCREMENT = auto()