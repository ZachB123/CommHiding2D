from mpi4py import MPI
import numpy as np
from composed_gemm import Gemm1D, Gemm2D, MatrixCommunicated, SubtileScheme, CommunicationDirection

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# min dims: m%PY, k%SIZE, n%PX → m%3, k%6, n%2 (px=2, py=3)
m, k, n, px, py = 6, 6, 2, 2, 3

alg = Gemm2D(
    Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV),
    Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
)
result = alg.setup_and_run(m, k, n, px, py)
if rank == 0:
    print(f"correct={result['correct']}")
    if not result['correct']:
        print(f"expected:\n{result['expected']}")
        print(f"actual:\n{result['actual']}")
