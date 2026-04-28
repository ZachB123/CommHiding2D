from mpi4py import MPI
import numpy as np
from distribution import row_major_distribution, col_major_distribution, pure_row_distribution
from distribution import get_subtile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Tiny case: m=6, k=4, n=3, px=3, py=2 (all divisible)
m, k, n, px, py = 6, 4, 3, 3, 2
size = 6

np.random.seed(42)
A = np.random.randint(-9, 10, size=(m, k)).astype(np.float32)
B = np.random.randint(-9, 10, size=(k, n)).astype(np.float32)
C = np.random.randint(-9, 10, size=(m, n)).astype(np.float32)
expected = np.matmul(A, B) + C

A_local = row_major_distribution(A, px, py, rank)
B_local = col_major_distribution(B, py, px, rank)
C_local = pure_row_distribution(C, size, rank)

if rank == 0:
    print(f"A_local shapes: {A_local.shape}")
    print(f"B_local shapes: {B_local.shape}")
    print(f"C_local shapes: {C_local.shape}")
    print(f"A=\n{A}")
    print(f"B=\n{B}")
    print(f"expected C=\n{expected}")

comm.Barrier()
for r in range(6):
    if rank == r:
        print(f"rank {r}: A_local={A_local}, B_local={B_local}, C_local={C_local}")
    comm.Barrier()
