import numpy as np
from mpi4py import MPI

DEBUG_RANK = 0

MATRIX_DTYPE = np.float32
MPI_DTYPE = MPI.FLOAT if MATRIX_DTYPE == np.float32 else MPI.DOUBLE
