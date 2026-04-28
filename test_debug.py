from mpi4py import MPI
import numpy as np
from composed_gemm import Gemm1D, Gemm2D, MatrixCommunicated, SubtileScheme, CommunicationDirection

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

m, k, n, px, py = 6, 98, 33, 3, 2

alg = Gemm2D(
    Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV),
    Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
)
result = alg.setup_and_run(m, k, n, px, py)
if rank == 0:
    print(f"correct={result['correct']}")
    if not result['correct']:
        print(f"expected[:3,:5]={result['expected'][:3,:5]}")
        print(f"actual[:3,:5]={result['actual'][:3,:5]}")
