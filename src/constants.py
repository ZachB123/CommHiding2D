import os
from pathlib import Path
import numpy as np
from mpi4py import MPI

DEBUG_RANK = 0

MATRIX_DTYPE = np.float32
MPI_DTYPE = MPI.FLOAT if MATRIX_DTYPE == np.float32 else MPI.DOUBLE

SRC_DIR = Path(__file__).parents[0].resolve()
PROJECT_ROOT = SRC_DIR.parent
BENCHMARK_FOLDER = PROJECT_ROOT / "benchmarks"