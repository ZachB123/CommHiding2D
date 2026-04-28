from mpi4py import MPI
import numpy as np
from composed_gemm import Gemm1D, Gemm2D, MatrixCommunicated, SubtileScheme, CommunicationDirection, GEMM_2D_INNER_CONFIGS, _make_compute_fn
from distribution import row_major_distribution, col_major_distribution, pure_row_distribution, get_subtile
from communicator import nearby_rank_communicator, remainder_communicator
from util import mpi_setup, generate_matrices, DoubleBuffer
from constants import MATRIX_DTYPE
from enums import SubtileScheme as SS, MatrixCommunicated as MC

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

m, k, n, px, py = 6, 4, 3, 3, 2
size = comm.Get_size()

np.random.seed(42)
A = np.random.randint(-9, 10, size=(m, k)).astype(np.float32)
B = np.random.randint(-9, 10, size=(k, n)).astype(np.float32)
C = np.random.randint(-9, 10, size=(m, n)).astype(np.float32)

# group_param_is_py = True (py)
group_param = py

# RS outer: nearby=outer, remainder=inner
outer_comm = nearby_rank_communicator(comm, group_param, rank)
inner_comm = remainder_communicator(comm, group_param, rank)

outer_size = outer_comm.Get_size()
inner_size = inner_comm.Get_size()
outer_rank = outer_comm.Get_rank()
inner_rank = inner_comm.Get_rank()

A_local = row_major_distribution(A, px, py, rank)
B_local = col_major_distribution(B, py, px, rank)
C_local = pure_row_distribution(C, size, rank)

comm.Barrier()
for r in range(6):
    if rank == r:
        print(f"rank {r}: outer_rank={outer_rank} outer_size={outer_size} inner_rank={inner_rank} inner_size={inner_size}")
        print(f"  A_local={A_local.flatten()}, B_local={B_local.flatten()}, C_local={C_local.flatten()}")
    comm.Barrier()

# Manually run outer step by step
from util import send, receive, AccumulationBuffer, call_algorithm
from enums import CommunicationDirection as CD

buffer = AccumulationBuffer(C_local)
direction = CD.SEND_PREV
direction_increment = 1
outer_index = (outer_rank + direction_increment) % outer_size

from composed_gemm import _A, _B, _C, _COL, _ROW, GEMM_2D_INNER_CONFIGS
from composed_gemm import CurrentTiles
config = GEMM_2D_INNER_CONFIGS[(_C, _ROW), (_B, _COL)]

inner_gemm1d = Gemm1D(MC.B, SS.COL, CD.SEND_PREV)

C_result = C_local.copy()

for i in range(outer_size):
    oi = outer_index
    tiles = config['tiles'](A_local, B_local, oi, outer_size)
    set_c_fn = config['set_c'](A_local, B_local, oi, outer_size) if config['set_c'] else None
    C_inner = config['make_C_inner'](C_local, oi, inner_size, outer_size) if config['make_C_inner'] else C_result
    
    result_inner = inner_gemm1d.run(
        A_local, B_local, C_inner, inner_comm, inner_size, inner_rank,
        current_tiles_override=tiles,
        set_c_override=set_c_fn,
        buffer_override=None,
        loopback=False,
    )
    
    if rank == 0 or rank == 1:
        print(f"rank {rank}, outer step i={i}, oi={oi}: result_inner={result_inner.flatten()}")
    
    if i == 0:
        C_curr = np.zeros(C_local.shape, dtype=MATRIX_DTYPE)
    else:
        from mpi4py import MPI as mpi_mod
        mpi_mod.Request.Waitall([receive_request, send_request])
        C_curr = buffer.buffer.copy()
    
    C_tmp = result_inner + C_curr
    
    if i == outer_size - 1:
        C_result = C_result + C_tmp
    else:
        from util import send as send_fn, MPI_DTYPE
        send_request, receive_request = send_fn(outer_comm, buffer.get_send_tile(C_tmp), buffer.get_receive_tile(), direction=direction)
    
    outer_index = (outer_index + direction_increment) % outer_size

comm.Barrier()
for r in range(6):
    if rank == r:
        print(f"rank {r}: final C_result={C_result.flatten()}")
    comm.Barrier()
