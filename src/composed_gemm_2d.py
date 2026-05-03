import functools
import numpy as np

from constants import MATRIX_DTYPE
from data_classes import CurrentTiles, DistributionFunctions, DivisibiltyRequirements, Gemm2DAlgorithmConfiguration, Gemm2DInnerLoopConfiguration
from debug import print_full_matrices, rank_print
from communicator import nearby_rank_communicator, remainder_communicator
from distribution import (get_subtile, get_subtile_shape, set_subtile,
    pure_column_distribution, pure_column_distribution_get_local_indices,
    pure_row_distribution, pure_row_distribution_get_local_indices,
    row_major_distribution, row_major_distribution_get_local_indices,
    col_major_distribution, col_major_distribution_get_local_indices,
    block_cyclic_distribution, alternating_column_distribution,
    alternating_row_distribution, A9_distribution, C11_get_local_indices)
from enums import CommunicationDirection, GemmDimension, MatrixCommunicated, SubtileScheme
from util import AccumulationBuffer, DoubleBuffer, SubtileBuffer, assemble_matrix_from_tiles, call_algorithm, create_algorithm_output, generate_matrices, matrices_equal, mpi_setup, receive, send
from composed_gemm_1d import Gemm1D


def noop_set_c(C, C_tmp, index, size):
    pass


# Outer current_tiles for 2D return full matrices, let inner loop handle subtiling
GEMM_2D_OUTER_LOOP_CURRENT_TILES = {
    MatrixCommunicated.A: CurrentTiles(
        lambda A, B, C, buffer, index, size: buffer.get_buffer(),
        lambda A, B, C, buffer, index, size: B,
        lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
    MatrixCommunicated.B: CurrentTiles(
        lambda A, B, C, buffer, index, size: A,
        lambda A, B, C, buffer, index, size: buffer.get_buffer(),
        lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
    MatrixCommunicated.C: CurrentTiles(
        lambda A, B, C, buffer, index, size: A,
        lambda A, B, C, buffer, index, size: B,
        lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
}

# Parameterized 2D inner loop configs
# First element = outer loop type, second = inner loop type.
GEMM_2D_INNER_LOOP_CONFIGURATIONS = {
    # Alg 1: outer=AG_A_COL, inner=AG_A_ROW
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(A_outer), make_contiguous=False),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: get_subtile(C, outer_size, 1, outer_index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, outer_size, 1, outer_index, 0),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 2: outer=AG_A_COL, inner=AG_B_COL
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(B, outer_size, 1, outer_index, 0),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 3: outer=AG_B_ROW, inner=AG_A_COL
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(A, 1, outer_size, 0, outer_index, needs_contiguous=True),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B_outer, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: C,
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: np.copyto(C, C_tmp),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 4: outer=RS_C_COL, inner=AG_A_COL
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(A, make_contiguous=True),
            persistent_buffer=True,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B, size, outer_size, index, outer_index),
                lambda A, B, C, buffer, index, size: C,
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: np.copyto(C, C_tmp),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 5: outer=RS_C_ROW, inner=AG_A_COL
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(A, outer_size, 1, outer_index, 0, needs_contiguous=True),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: C,
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: np.copyto(C, C_tmp),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 6: outer=AG_A_ROW, inner=AG_B_COL
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(B, make_contiguous=False),
            persistent_buffer=True,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, outer_size, size, outer_index, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, outer_size, size, outer_index, index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 7: outer=AG_B_ROW, inner=AG_A_ROW
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(A, 1, outer_size, 0, outer_index, needs_contiguous=True),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: B_outer,
                lambda A, B, C, buffer, index, size: get_subtile(C, size, 1, index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, 1, index, 0),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 8: outer=AG_A_ROW, inner=RS_C_COL
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: get_subtile(B, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(get_subtile_shape(C, outer_size, 1), dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                set_subtile(C, result + get_subtile(C, outer_size, 1, outer_index, 0), outer_size, 1, outer_index, 0),
        ),
    # Alg 9: outer=RS_C_ROW, inner=AG_A_ROW
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(A, outer_size, 1, outer_index, 0),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: B,
                lambda A, B, C, buffer, index, size: get_subtile(C, size, 1, index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, 1, index, 0),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 10: outer=AG_B_COL, inner=AG_B_ROW
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(B_outer), make_contiguous=False),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, outer_size, 0, outer_index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, outer_size, 0, outer_index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # Alg 11: outer=RS_C_COL, inner=AG_B_COL
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(B, 1, outer_size, 0, outer_index, needs_contiguous=True),
            persistent_buffer=False,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A,
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 12: outer=AG_B_COL, inner=RS_C_ROW
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: B_outer,
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(get_subtile_shape(C, 1, outer_size), dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                set_subtile(C, result + get_subtile(C, 1, outer_size, 0, outer_index), 1, outer_size, 0, outer_index),
        ),
    # Alg 13: outer=RS_C_COL, inner=AG_B_ROW
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(B, 1, outer_size, 0, outer_index, needs_contiguous=True),
            persistent_buffer=False,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: C,
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: np.copyto(C, C_tmp),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 14: outer=RS_C_ROW, inner=AG_B_ROW
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(B, make_contiguous=False),
            persistent_buffer=True,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, outer_size, size, outer_index, index),
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: C,
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: np.copyto(C, C_tmp),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # Alg 15: outer=RS_C_COL, inner=RS_C_ROW
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, outer_size, 1, outer_index, 0),
                lambda A, B, C, buffer, index, size: get_subtile(B, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),

    # === Reversed algorithm inner configs ===
    # R1: outer=AG_A_ROW, inner=AG_A_COL (reverse of Alg 1)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(A_outer), make_contiguous=False),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: get_subtile(C, outer_size, 1, outer_index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, outer_size, 1, outer_index, 0),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R2: outer=AG_B_COL, inner=AG_A_COL (reverse of Alg 2)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(A, 1, outer_size, 0, outer_index, needs_contiguous=True),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: B_outer,
                lambda A, B, C, buffer, index, size: get_subtile(C, size, 1, index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, 1, index, 0),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R3: outer=AG_A_COL, inner=AG_B_ROW (reverse of Alg 3)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(B, outer_size, 1, outer_index, 0),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R4: outer=AG_A_COL, inner=RS_C_COL (reverse of Alg 4)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: get_subtile(B, outer_size, size, outer_index, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                np.copyto(C, result + C),
        ),
    # R5: outer=AG_A_COL, inner=RS_C_ROW (reverse of Alg 5)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A_outer, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: get_subtile(B, outer_size, 1, outer_index, 0),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                np.copyto(C, result + C),
        ),
    # R6: outer=AG_B_COL, inner=AG_A_ROW (reverse of Alg 6)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(A), make_contiguous=False),
            persistent_buffer=True,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: B_outer,
                lambda A, B, C, buffer, index, size: get_subtile(C, size, outer_size, index, outer_index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, outer_size, index, outer_index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R7: outer=AG_A_ROW, inner=AG_B_ROW (reverse of Alg 7)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                SubtileBuffer(B, outer_size, 1, outer_index, 0),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A_outer,
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R8: outer=RS_C_COL, inner=AG_A_ROW (reverse of Alg 8)
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(A, make_contiguous=True),
            persistent_buffer=True,
            loopback=True,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(B, 1, outer_size, 0, outer_index),
                lambda A, B, C, buffer, index, size: get_subtile(C, size, 1, index, 0),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, 1, index, 0),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # R9: outer=AG_A_ROW, inner=RS_C_ROW (reverse of Alg 9)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A_outer, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: B,
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(get_subtile_shape(C, outer_size, 1), dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                set_subtile(C, result + get_subtile(C, outer_size, 1, outer_index, 0), outer_size, 1, outer_index, 0),
        ),
    # R10: outer=AG_B_ROW, inner=AG_B_COL (reverse of Alg 10)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(B_outer), make_contiguous=False),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, 1, outer_size, 0, outer_index),
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=None,
            reduce_scatter_finalize=None,
        ),
    # R11: outer=AG_B_COL, inner=RS_C_COL (reverse of Alg 11)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: A,
                lambda A, B, C, buffer, index, size: get_subtile(B_outer, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(get_subtile_shape(C, 1, outer_size), dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                set_subtile(C, result + get_subtile(C, 1, outer_size, 0, outer_index), 1, outer_size, 0, outer_index),
        ),
    # R12: outer=RS_C_ROW, inner=AG_B_COL (reverse of Alg 12)
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=lambda A, B, C, A_outer, B_outer, outer_index, inner_size, outer_size:
                DoubleBuffer(np.copy(B), make_contiguous=False),
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, outer_size, 1, outer_index, 0),
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index),
            ),
            set_c_tile=lambda A_outer, B_outer, outer_index, outer_size:
                lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
    # R13: outer=AG_B_ROW, inner=RS_C_COL (reverse of Alg 13)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, 1, outer_size, 0, outer_index),
                lambda A, B, C, buffer, index, size: get_subtile(B_outer, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                np.copyto(C, result + C),
        ),
    # R14: outer=AG_B_ROW, inner=RS_C_ROW (reverse of Alg 14)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, size, outer_size, index, outer_index),
                lambda A, B, C, buffer, index, size: B_outer,
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=lambda result, C, outer_index, inner_size, outer_size:
                np.copyto(C, result + C),
        ),
    # R15: outer=RS_C_ROW, inner=RS_C_COL (reverse of Alg 15)
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DInnerLoopConfiguration(
            make_buffer=None,
            persistent_buffer=False,
            loopback=False,
            tiles=lambda A_outer, B_outer, outer_index, outer_size: CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, outer_size, 1, outer_index, 0),
                lambda A, B, C, buffer, index, size: get_subtile(B, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(C.shape, dtype=MATRIX_DTYPE),
            ),
            set_c_tile=None,
            make_inner_c_matrix=lambda C, outer_index, inner_size, outer_size:
                np.zeros(C.shape, dtype=MATRIX_DTYPE),
            reduce_scatter_finalize=None,
        ),
}


def make_inner_loop_compute_function(algorithm_key, A, B, C, inner_comm, inner_size, inner_rank,
                                     outer_size, px, py, inner_config, outer_config, skip_computation=False):
    config = GEMM_2D_INNER_LOOP_CONFIGURATIONS[algorithm_key]
    outer_is_all_gather = outer_config.matrix_communicated != MatrixCommunicated.C

    inner_gemm1d = Gemm1D(inner_config.matrix_communicated,
                          inner_config.subtile_scheme,
                          inner_config.communication_direction)

    persistent_buf = None
    if config.persistent_buffer and config.make_buffer:
        persistent_buf = config.make_buffer(A, B, C, None, None, None, inner_size, outer_size)

    def compute_function(A_outer, B_outer, i, outer_index):
        is_last_outer = (i == outer_size - 1)
        loopback = config.loopback and not is_last_outer

        if persistent_buf is not None:
            buffer = persistent_buf
        elif config.make_buffer:
            buffer = config.make_buffer(A, B, C, A_outer, B_outer,
                                        outer_index, inner_size, outer_size)
        else:
            buffer = None

        tiles = config.tiles(A_outer, B_outer, outer_index, outer_size)
        set_c_tile_function = config.set_c_tile(A_outer, B_outer, outer_index, outer_size) if config.set_c_tile else None

        if config.make_inner_c_matrix:
            C_inner = config.make_inner_c_matrix(C, outer_index, inner_size, outer_size)
        else:
            C_inner = C

        result = inner_gemm1d.run(
            A, B, C_inner, inner_comm, inner_size, inner_rank,
            current_tiles_override=tiles,
            set_c_override=set_c_tile_function,
            buffer_override=buffer,
            loopback=loopback,
            skip_computation=skip_computation,
        )

        if config.make_inner_c_matrix:
            if outer_is_all_gather:
                config.reduce_scatter_finalize(result, C, outer_index, inner_size, outer_size)
                return np.zeros(C.shape, dtype=MATRIX_DTYPE)
            else:
                return result
        else:
            return np.zeros(C.shape, dtype=MATRIX_DTYPE)

    return compute_function


GEMM_2D_ALGORITHM_CONFIGURATIONS = {
    # Alg 1: outer=AG_A_COL, inner=AG_A_ROW
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # Alg 2: outer=AG_A_COL, inner=AG_B_COL
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),
    # Alg 3: outer=AG_B_ROW, inner=AG_A_COL
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, px, py, outer_comm.Get_rank(), inner_comm.Get_rank()),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(py, rank)),
        ),
    # Alg 4: outer=RS_C_COL, inner=AG_A_COL
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_column_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # Alg 5: outer=RS_C_ROW, inner=AG_A_COL
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_column_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),
    # Alg 6: outer=AG_A_ROW, inner=AG_B_COL
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_row_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(py, rank)),
        ),
    # Alg 7: outer=AG_B_ROW, inner=AG_A_ROW
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(py, rank)),
        ),
    # Alg 8: outer=AG_A_ROW, inner=RS_C_COL
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # Alg 9: outer=RS_C_ROW, inner=AG_A_ROW
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: A9_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),
    # Alg 10: outer=AG_B_COL, inner=AG_B_ROW
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # Alg 11: outer=RS_C_COL, inner=AG_B_COL
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, px, py, inner_comm.Get_rank(), outer_comm.Get_rank()),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: C11_get_local_indices(C, px, py, inner_comm.Get_rank(), outer_comm.Get_rank()),
            flatten_gather=True,
        ),
    # Alg 12: outer=AG_B_COL, inner=RS_C_ROW
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # Alg 13: outer=RS_C_COL, inner=AG_B_ROW
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, px, py, inner_comm.Get_rank(), outer_comm.Get_rank()),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(py, rank)),
        ),
    # Alg 14: outer=RS_C_ROW, inner=AG_B_ROW
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, py, px, inner_comm.Get_rank(), outer_comm.Get_rank()),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # Alg 15: outer=RS_C_COL, inner=RS_C_ROW
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),

    # Reversed algorithm entries
    # R1: outer=AG_A_ROW, inner=AG_A_COL (reverse of Alg 1)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # R2: outer=AG_B_COL, inner=AG_A_COL (reverse of Alg 2)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(px, rank)),
        ),
    # R3: outer=AG_A_COL, inner=AG_B_ROW (reverse of Alg 3)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),
    # R4: outer=AG_A_COL, inner=RS_C_COL (reverse of Alg 4)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_column_distribution(M, py, px, rank).copy(),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # R5: outer=AG_A_COL, inner=RS_C_ROW (reverse of Alg 5)
    ((MatrixCommunicated.A, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_column_distribution(M, px, py, rank).copy(),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(py, rank)),
        ),
    # R6: outer=AG_B_COL, inner=AG_A_ROW (reverse of Alg 6)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: alternating_column_distribution(M, py, px, rank).copy(),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(px, rank)),
        ),
    # R7: outer=AG_A_ROW, inner=AG_B_ROW (reverse of Alg 7)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(px, rank)),
        ),
    # R8: outer=RS_C_COL, inner=AG_A_ROW (reverse of Alg 8)
    ((MatrixCommunicated.C, SubtileScheme.COL), (MatrixCommunicated.A, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_column_distribution_get_local_indices(rank)),
        ),
    # R9: outer=AG_A_ROW, inner=RS_C_ROW (reverse of Alg 9)
    ((MatrixCommunicated.A, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: A9_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(py, rank)),
        ),
    # R10: outer=AG_B_ROW, inner=AG_B_COL (reverse of Alg 10)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # R11: outer=AG_B_COL, inner=RS_C_COL (reverse of Alg 11)
    ((MatrixCommunicated.B, SubtileScheme.COL), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, py, px, outer_comm.Get_rank(), inner_comm.Get_rank()),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: C11_get_local_indices(C, py, px, outer_comm.Get_rank(), inner_comm.Get_rank()),
            flatten_gather=True,
        ),
    # R12: outer=RS_C_ROW, inner=AG_B_COL (reverse of Alg 12)
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.B, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, px, py, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # R13: outer=AG_B_ROW, inner=RS_C_COL (reverse of Alg 13)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: px,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, py, px, outer_comm.Get_rank(), inner_comm.Get_rank()),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: row_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, row_major_distribution_get_local_indices(px, rank)),
        ),
    # R14: outer=AG_B_ROW, inner=RS_C_ROW (reverse of Alg 14)
    ((MatrixCommunicated.B, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.ROW)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: block_cyclic_distribution(M, px, py, outer_comm.Get_rank(), inner_comm.Get_rank()),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, pure_row_distribution_get_local_indices(rank)),
        ),
    # R15: outer=RS_C_ROW, inner=RS_C_COL (reverse of Alg 15)
    ((MatrixCommunicated.C, SubtileScheme.ROW), (MatrixCommunicated.C, SubtileScheme.COL)):
        Gemm2DAlgorithmConfiguration(
            group_param=lambda px, py: py,
            divisibility=DivisibiltyRequirements(GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
            distribution=DistributionFunctions(
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_column_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: pure_row_distribution(M, size, rank),
                lambda M, px, py, rank, size, outer_comm, inner_comm: col_major_distribution(M, py, px, rank),
            ),
            get_local_indices=lambda C, rank, px, py, size, outer_comm, inner_comm: (C, col_major_distribution_get_local_indices(py, rank)),
        ),
}


class Gemm2DConfig:
    def __init__(self, algorithm_key):
        self.algorithm_key = algorithm_key
        config = GEMM_2D_ALGORITHM_CONFIGURATIONS[algorithm_key]
        self.group_param = config.group_param
        self.divisibility = config.divisibility
        self.distribution = config.distribution
        self.get_local_indices = config.get_local_indices
        self.flatten_gather = config.flatten_gather

    def assert_divisibility(self, m, k, n, px, py, size):
        def check(gemm_dim, value, name):
            if gemm_dim == GemmDimension.ONE:
                assert value >= 1, f"{name}={value} must be >= 1"
            elif gemm_dim == GemmDimension.PX:
                assert value % px == 0, f"{name}={value} not divisible by px={px}"
            elif gemm_dim == GemmDimension.PY:
                assert value % py == 0, f"{name}={value} not divisible by py={py}"
            elif gemm_dim == GemmDimension.SIZE:
                assert value % size == 0, f"{name}={value} not divisible by size={size}"
            else:
                raise ValueError(f"Invalid GemmDimension: {gemm_dim}")
        check(self.divisibility.m_divisibility, m, "m")
        check(self.divisibility.k_divisibility, k, "k")
        check(self.divisibility.n_divisibility, n, "n")

    def __repr__(self):
        return (
            "Gemm2DConfig(\n"
            f"  algorithm_key={self.algorithm_key},\n"
            f"  divisibility={self.divisibility},\n"
            f"  flatten_gather={self.flatten_gather}\n"
            ")"
        )


class Gemm2D:
    def __init__(self, first, second):
        outer_key = (first.config.matrix_communicated, first.config.subtile_scheme)
        inner_key = (second.config.matrix_communicated, second.config.subtile_scheme)
        key = (outer_key, inner_key)

        if key not in GEMM_2D_ALGORITHM_CONFIGURATIONS:
            raise ValueError(f"No 2D algorithm for combination: outer={outer_key}, inner={inner_key}")

        self.algorithm_key = key
        self.config = Gemm2DConfig(key)
        self.outer_gemm1d = first
        self.inner_gemm1d = second

    def setup_and_run(self, m, k, n, px, py, skip_computation=False):
        comm, size, rank = mpi_setup()

        self.config.assert_divisibility(m, k, n, px, py, size)

        A, B, C = generate_matrices(m, k, n)
        expected = np.matmul(A, B) + C

        group_param = self.config.group_param(px, py)
        if self.outer_gemm1d.config.matrix_communicated != MatrixCommunicated.C:
            outer_comm = remainder_communicator(comm, group_param, rank)
            inner_comm = nearby_rank_communicator(comm, group_param, rank)
        else:
            outer_comm = nearby_rank_communicator(comm, group_param, rank)
            inner_comm = remainder_communicator(comm, group_param, rank)

        A_local = self.config.distribution.A_distribution(A, px, py, rank, size, outer_comm, inner_comm)
        B_local = self.config.distribution.B_distribution(B, px, py, rank, size, outer_comm, inner_comm)
        C_local = self.config.distribution.C_distribution(C, px, py, rank, size, outer_comm, inner_comm)

        run_function = functools.partial(self.run, skip_computation=skip_computation)
        C_local, elapsed_time = call_algorithm(
            run_function, comm, A_local, B_local, C_local,
            outer_comm, inner_comm, px, py
        )

        gather_result = self.config.get_local_indices(C_local, rank, px, py, size, outer_comm, inner_comm)
        if self.config.flatten_gather:
            actual_tiles_raw = comm.allgather(gather_result)
            actual_tiles = [item for sublist in actual_tiles_raw for item in sublist]
        else:
            actual_tiles = comm.allgather(gather_result)

        actual = assemble_matrix_from_tiles(actual_tiles)
        correct = matrices_equal(expected, actual)

        return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)

    def run(self, A, B, C, outer_comm, inner_comm, px, py, skip_computation=False):
        outer_size = outer_comm.Get_size()
        outer_rank = outer_comm.Get_rank()
        inner_size = inner_comm.Get_size()
        inner_rank = inner_comm.Get_rank()

        compute_function = make_inner_loop_compute_function(
            self.algorithm_key,
            A, B, C, inner_comm, inner_size, inner_rank,
            outer_size, px, py,
            self.inner_gemm1d.config,
            self.outer_gemm1d.config,
            skip_computation=skip_computation,
        )

        outer_matrix_communicated = self.outer_gemm1d.config.matrix_communicated
        outer_current_tiles = GEMM_2D_OUTER_LOOP_CURRENT_TILES[outer_matrix_communicated]

        return self.outer_gemm1d.run(
            A, B, C, outer_comm, outer_size, outer_rank,
            compute_function=compute_function,
            current_tiles_override=outer_current_tiles,
            set_c_override=noop_set_c,
            skip_computation=skip_computation,
        )
