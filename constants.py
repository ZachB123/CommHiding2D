import numpy as np
from mpi4py import MPI


MATRIX_DTYPE = np.float64
MPI_DTYPE = MPI.FLOAT if MATRIX_DTYPE == np.float32 else MPI.DOUBLE
