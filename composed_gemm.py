from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from constants import MATRIX_DTYPE
from data_classes import CurrentTiles, DistributionFunctions, DivisibiltyRequirements
from debug import print_full_matrices, rank_print
from communicator import nearby_rank_communicator, remainder_communicator
from distribution import (get_subtile, get_subtile_shape, set_subtile,
    pure_column_distribution, pure_column_distribution_get_local_indices,
    pure_row_distribution, pure_row_distribution_get_local_indices,
    row_major_distribution, row_major_distribution_get_local_indices,
    col_major_distribution, col_major_distribution_get_local_indices,
    block_cyclic_distribution, alternating_column_distribution,
    alternating_row_distribution, A9_distribution, C11_get_local_indices)
from enums import CommunicationDirection, ConfigurationOptions1D, GemmDimension, MatrixCommunicated, SubtileScheme
from util import AccumulationBuffer, DoubleBuffer, SubtileBuffer, assemble_matrix_from_tiles, call_algorithm, create_algorithm_output, generate_matrices, matrices_equal, mpi_setup, receive, send

# when multiple items needed to index specify in this order (MatrixCommunicated, SubtileScheme, CommunicationDirection)
GEMM_1D_CONFIGURATIONS = {
    ConfigurationOptions1D.DIVISIBILITY: {
        (MatrixCommunicated.A, SubtileScheme.COL):
            DivisibiltyRequirements(GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
        (MatrixCommunicated.A, SubtileScheme.ROW):
            DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
        (MatrixCommunicated.B, SubtileScheme.COL):
            DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
        (MatrixCommunicated.B, SubtileScheme.ROW):
            DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
        (MatrixCommunicated.C, SubtileScheme.COL):
            DivisibiltyRequirements(GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
        (MatrixCommunicated.C, SubtileScheme.ROW):
            DivisibiltyRequirements(GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    },
    ConfigurationOptions1D.DISTRIBUTION: {
        (MatrixCommunicated.A, SubtileScheme.COL):
            DistributionFunctions(pure_column_distribution, pure_column_distribution, pure_column_distribution),
        (MatrixCommunicated.A, SubtileScheme.ROW):
            DistributionFunctions(pure_row_distribution, pure_column_distribution, pure_column_distribution),
        (MatrixCommunicated.B, SubtileScheme.COL):
            DistributionFunctions(pure_row_distribution, pure_column_distribution, pure_row_distribution),
        (MatrixCommunicated.B, SubtileScheme.ROW):
            DistributionFunctions(pure_row_distribution, pure_row_distribution, pure_row_distribution),
        (MatrixCommunicated.C, SubtileScheme.COL):
            DistributionFunctions(pure_column_distribution, pure_row_distribution, pure_column_distribution),
        (MatrixCommunicated.C, SubtileScheme.ROW):
            DistributionFunctions(pure_column_distribution, pure_row_distribution, pure_row_distribution),
    },
    ConfigurationOptions1D.GET_LOCAL_INDICES: {
        (MatrixCommunicated.A, SubtileScheme.COL):
            pure_column_distribution_get_local_indices,
        (MatrixCommunicated.A, SubtileScheme.ROW):
            pure_column_distribution_get_local_indices,
        (MatrixCommunicated.B, SubtileScheme.COL):
            pure_row_distribution_get_local_indices,
        (MatrixCommunicated.B, SubtileScheme.ROW):
            pure_row_distribution_get_local_indices,
        (MatrixCommunicated.C, SubtileScheme.COL):
            pure_column_distribution_get_local_indices,
        (MatrixCommunicated.C, SubtileScheme.ROW):
            pure_row_distribution_get_local_indices,
    },
    # actual algorithm configurations
    ConfigurationOptions1D.INDEX: {
        MatrixCommunicated.A: lambda rank, size, direction_increment: rank,
        MatrixCommunicated.B: lambda rank, size, direction_increment: rank,
        MatrixCommunicated.C: lambda rank, size, direction_increment: (rank + direction_increment) % size,
    },
    ConfigurationOptions1D.BUFFER: {
        MatrixCommunicated.A: lambda A, B, C: DoubleBuffer(A, make_contiguous=False),
        MatrixCommunicated.B: lambda A, B, C: DoubleBuffer(B, make_contiguous=False),
        MatrixCommunicated.C: lambda A, B, C: AccumulationBuffer(C)
    },
    ConfigurationOptions1D.CURRENT_TILES: {
        (MatrixCommunicated.A, SubtileScheme.COL):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(), 
                lambda A, B, C, buffer, index, size: get_subtile(B, size, 1, index, 0),
                lambda A, B, C, buffer, index, size: C    
            ),
        (MatrixCommunicated.A, SubtileScheme.ROW):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: buffer.get_buffer(), 
                lambda A, B, C, buffer, index, size: B,
                lambda A, B, C, buffer, index, size: get_subtile(C, size, 1, index, 0)    
            ),
        (MatrixCommunicated.B, SubtileScheme.COL):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: A, 
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: get_subtile(C, 1, size, 0, index)    
            ),
        (MatrixCommunicated.B, SubtileScheme.ROW):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, 1, size, 0, index), 
                lambda A, B, C, buffer, index, size: buffer.get_buffer(),
                lambda A, B, C, buffer, index, size: C    
            ),
        (MatrixCommunicated.C, SubtileScheme.COL):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: A, 
                lambda A, B, C, buffer, index, size: get_subtile(B, 1, size, 0, index),
                lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)   
            ),
        (MatrixCommunicated.C, SubtileScheme.ROW):
            CurrentTiles(
                lambda A, B, C, buffer, index, size: get_subtile(A, size, 1, index, 0), 
                lambda A, B, C, buffer, index, size: B,
                lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE) 
            ),
    },
    ConfigurationOptions1D.SET_C: {
        (MatrixCommunicated.A, SubtileScheme.COL):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, 1, 0, 0),
        (MatrixCommunicated.A, SubtileScheme.ROW):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, size, 1, index, 0),
        (MatrixCommunicated.B, SubtileScheme.COL):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, size, 0, index),
        (MatrixCommunicated.B, SubtileScheme.ROW):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, 1, 0, 0),
        (MatrixCommunicated.C, SubtileScheme.COL):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, 1, 0, 0),
        (MatrixCommunicated.C, SubtileScheme.ROW):
            lambda C, C_tmp, index, size: set_subtile(C, C_tmp, 1, 1, 0, 0),
    },
    ConfigurationOptions1D.DIRECTION_INCREMENT: {
        CommunicationDirection.SEND_PREV: 1,
        CommunicationDirection.SEND_NEXT: -1
    }
}

class Gemm1DConfig:
    def __init__(
        self,
        matrix_communicated: MatrixCommunicated,
        subtile_scheme: SubtileScheme,
        communication_direction: CommunicationDirection,
    ):
        self.matrix_communicated = matrix_communicated
        self.subtile_scheme = subtile_scheme
        self.communication_direction = communication_direction

        self.divisibility = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.DIVISIBILITY][(self.matrix_communicated, self.subtile_scheme)]
        self.distribution = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.DISTRIBUTION][(self.matrix_communicated, self.subtile_scheme)]
        self.get_local_indices = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.GET_LOCAL_INDICES][(self.matrix_communicated, self.subtile_scheme)]

        self.index = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.INDEX][self.matrix_communicated]
        self.buffer = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.BUFFER][self.matrix_communicated]
        self.current_tiles = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.CURRENT_TILES][(self.matrix_communicated, self.subtile_scheme)]
        self.set_c = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.SET_C][(self.matrix_communicated, self.subtile_scheme)]
        self.direction_increment = GEMM_1D_CONFIGURATIONS[ConfigurationOptions1D.DIRECTION_INCREMENT][communication_direction]

    def assert_divisibility(self, size, m, k, n):
        def assert_single_divisibility(gemm_dimension: GemmDimension, value):
            if gemm_dimension == GemmDimension.ONE:
                assert value >= 1, f"value: {value}"
            elif gemm_dimension == GemmDimension.SIZE:
                assert value % size == 0, f"size: {size}, value: {value}"
            else:
                raise ValueError(f"Invalid GemmDimension for 1D Gemm: {gemm_dimension}.")
        assert_single_divisibility(self.divisibility.m_divisibility, m)
        assert_single_divisibility(self.divisibility.k_divisibility, k)
        assert_single_divisibility(self.divisibility.n_divisibility, n)

    def __repr__(self) -> str:
        return (
            "Gemm1DConfig(\n"
            f"  matrix_communicated={self.matrix_communicated.name},\n"
            f"  subtile_scheme={self.subtile_scheme.name},\n"
            f"  communication_direction={self.communication_direction.name},\n"
            f"  divisibility={self.divisibility},\n"
            f"  distribution={self.distribution},\n"
            f"  get_local_indices={self.get_local_indices.__name__}\n"
            ")"
        )


class Gemm1D:
    def __init__(self,
                 matrix_communicated: MatrixCommunicated,
                 subtile_scheme: SubtileScheme,
                 communication_direction: CommunicationDirection
                 ):
        
        self.config = Gemm1DConfig(matrix_communicated, subtile_scheme, communication_direction)

    def setup_and_run(self, m, k, n, px, py):
        comm, size, rank = mpi_setup()

        self.config.assert_divisibility(size, m, k ,n)

        A, B, C = generate_matrices(m, k, n)
        expected = np.matmul(A, B) + C

        A_local = self.config.distribution.A_distribution(A, size, rank)
        B_local = self.config.distribution.B_distribution(B, size, rank)
        C_local = self.config.distribution.C_distribution(C, size, rank)
    
        # print_full_matrices(A, B, C)

        C_local, elapsed_time = call_algorithm(self.run, comm, A_local, B_local, C_local, comm, size, rank)

        actual_tiles = comm.allgather((C_local, self.config.get_local_indices(rank)))
        actual = assemble_matrix_from_tiles(actual_tiles)

        correct = matrices_equal(expected, actual)

        return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)

    def run(self, A, B, C, comm, size, rank, compute_fn=None,
            current_tiles_override=None, set_c_override=None,
            buffer_override=None, loopback=False):
        index = self.config.index(rank, size, self.config.direction_increment)
        loop_iterations = size
        buffer = buffer_override if buffer_override is not None else self.config.buffer(A, B, C)

        current_tiles = current_tiles_override if current_tiles_override is not None else self.config.current_tiles
        set_c = set_c_override if set_c_override is not None else self.config.set_c

        for i in range(loop_iterations):
            if self.config.matrix_communicated != MatrixCommunicated.C:
                should_send = (i != loop_iterations - 1) or loopback
                if should_send:
                    send_request, receive_request = send(comm, buffer.get_send_tile(), buffer.get_receive_tile(), direction=self.config.communication_direction)

            A_curr = current_tiles.A_curr(A, B, C, buffer, index, size)
            B_curr = current_tiles.B_curr(A, B, C, buffer, index, size)

            if compute_fn is not None:
                local_result = compute_fn(A_curr, B_curr, i, index)
            else:
                local_result = np.matmul(A_curr, B_curr)

            if self.config.matrix_communicated != MatrixCommunicated.C or i == 0:
                C_curr = current_tiles.C_curr(A, B, C, buffer, index, size)
            else:
                C_curr = receive([receive_request, send_request], buffer.on_receive())

            C_tmp = local_result + C_curr

            if self.config.matrix_communicated != MatrixCommunicated.C:
                set_c(C, C_tmp, index, size)

                if should_send:
                    receive([send_request, receive_request], buffer.on_receive())
            else:
                if i == loop_iterations - 1:
                    C = C + C_tmp
                else:
                    send_request, receive_request = send(comm, buffer.get_send_tile(C_tmp), buffer.get_receive_tile(), direction=self.config.communication_direction)

            index = (index + self.config.direction_increment) % size

        return C


# === 2D GEMM ===

# Outer current_tiles for 2D: return full matrices, let inner loop handle subtiling
GEMM_2D_OUTER_CURRENT_TILES = {
    MatrixCommunicated.A: CurrentTiles(
        A_curr=lambda A, B, C, buffer, index, size: buffer.get_buffer(),
        B_curr=lambda A, B, C, buffer, index, size: B,
        C_curr=lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
    MatrixCommunicated.B: CurrentTiles(
        A_curr=lambda A, B, C, buffer, index, size: A,
        B_curr=lambda A, B, C, buffer, index, size: buffer.get_buffer(),
        C_curr=lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
    MatrixCommunicated.C: CurrentTiles(
        A_curr=lambda A, B, C, buffer, index, size: A,
        B_curr=lambda A, B, C, buffer, index, size: B,
        C_curr=lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
}


def _noop_set_c(C, C_tmp, index, size):
    pass


_A, _B, _C = MatrixCommunicated.A, MatrixCommunicated.B, MatrixCommunicated.C
_COL, _ROW = SubtileScheme.COL, SubtileScheme.ROW


# Parameterized 2D inner loop configs
# Keys are ordered tuples ((outer_matrix, outer_subtile), (inner_matrix, inner_subtile))
# First element = outer loop type, second = inner loop type.

GEMM_2D_INNER_CONFIGS = {
    # Alg 1: outer=AG_A_COL, inner=AG_A_ROW
    ((_A, _COL), (_A, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(A_outer), make_contiguous=False),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, sz, 1, idx, 0),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, os_, 1, oi, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, os_, 1, oi, 0),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 2: outer=AG_A_COL, inner=AG_B_COL
    ((_A, _COL), (_B, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(B, os_, 1, oi, 0),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 3: outer=AG_B_ROW, inner=AG_A_COL
    ((_B, _ROW), (_A, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(A, 1, os_, 0, oi, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B_outer, sz, 1, idx, 0),
            C_curr=lambda A, B, C, buf, idx, sz: C,
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: np.copyto(C, Ct),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 4: outer=RS_C_COL, inner=AG_A_COL
    ((_C, _COL), (_A, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(A, make_contiguous=True),
        'persistent_buffer': True,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, sz, os_, idx, oi),
            C_curr=lambda A, B, C, buf, idx, sz: C,
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: np.copyto(C, Ct),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 5: outer=RS_C_ROW, inner=AG_A_COL
    ((_C, _ROW), (_A, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(A, os_, 1, oi, 0, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, sz, 1, idx, 0),
            C_curr=lambda A, B, C, buf, idx, sz: C,
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: np.copyto(C, Ct),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 6: outer=AG_A_ROW, inner=AG_B_COL
    ((_A, _ROW), (_B, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(B, make_contiguous=False),
        'persistent_buffer': True,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, os_, sz, oi, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, os_, sz, oi, idx),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 7: outer=AG_B_ROW, inner=AG_A_ROW
    ((_B, _ROW), (_A, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(A, 1, os_, 0, oi, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: B_outer,
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, sz, 1, idx, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, sz, 1, idx, 0),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 8: outer=AG_A_ROW, inner=RS_C_COL
    ((_A, _ROW), (_C, _COL)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, 1, sz, 0, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(get_subtile_shape(C, os_, 1), dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            set_subtile(C, result + get_subtile(C, os_, 1, oi, 0), os_, 1, oi, 0),
    },
    # Alg 9: outer=RS_C_ROW, inner=AG_A_ROW
    ((_C, _ROW), (_A, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(A, os_, 1, oi, 0),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: B,
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, sz, 1, idx, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, sz, 1, idx, 0),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 10: outer=AG_B_COL, inner=AG_B_ROW
    ((_B, _COL), (_B, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(B_outer), make_contiguous=False),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, 1, sz, 0, idx),
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, os_, 0, oi),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, os_, 0, oi),
        'make_C_inner': None,
        'rs_final': None,
    },
    # Alg 11: outer=RS_C_COL, inner=AG_B_COL
    ((_C, _COL), (_B, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(B, 1, os_, 0, oi, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A,
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 12: outer=AG_B_COL, inner=RS_C_ROW
    ((_B, _COL), (_C, _ROW)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, sz, 1, idx, 0),
            B_curr=lambda A, B, C, buf, idx, sz: B_outer,
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(get_subtile_shape(C, 1, os_), dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            set_subtile(C, result + get_subtile(C, 1, os_, 0, oi), 1, os_, 0, oi),
    },
    # Alg 13: outer=RS_C_COL, inner=AG_B_ROW
    ((_C, _COL), (_B, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(B, 1, os_, 0, oi, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, 1, sz, 0, idx),
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: C,
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: np.copyto(C, Ct),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 14: outer=RS_C_ROW, inner=AG_B_ROW
    ((_C, _ROW), (_B, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(B, make_contiguous=False),
        'persistent_buffer': True,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, os_, sz, oi, idx),
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: C,
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: np.copyto(C, Ct),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # Alg 15: outer=RS_C_COL, inner=RS_C_ROW
    ((_C, _COL), (_C, _ROW)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, os_, 1, oi, 0),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, 1, sz, 0, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },

    # === Reversed algorithm inner configs ===
    # R1: outer=AG_A_ROW, inner=AG_A_COL (reverse of Alg 1)
    ((_A, _ROW), (_A, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(A_outer), make_contiguous=False),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, sz, 1, idx, 0),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, os_, 1, oi, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, os_, 1, oi, 0),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R10: outer=AG_B_ROW, inner=AG_B_COL (reverse of Alg 10)
    ((_B, _ROW), (_B, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(B_outer), make_contiguous=False),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, 1, os_, 0, oi),
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R6: outer=AG_B_COL, inner=AG_A_ROW (reverse of Alg 6)
    ((_B, _COL), (_A, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(A), make_contiguous=False),
        'persistent_buffer': True,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: B_outer,
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, sz, os_, idx, oi),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, sz, os_, idx, oi),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R15: outer=RS_C_ROW, inner=RS_C_COL (reverse of Alg 15)
    ((_C, _ROW), (_C, _COL)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, os_, 1, oi, 0),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, 1, sz, 0, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },

    # R3: outer=AG_A_COL, inner=AG_B_ROW (reverse of Alg 3)
    ((_A, _COL), (_B, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(B, os_, 1, oi, 0),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R2: outer=AG_B_COL, inner=AG_A_COL (reverse of Alg 2)
    ((_B, _COL), (_A, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(A, 1, os_, 0, oi, needs_contiguous=True),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: B_outer,
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, sz, 1, idx, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, sz, 1, idx, 0),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R7: outer=AG_A_ROW, inner=AG_B_ROW (reverse of Alg 7)
    ((_A, _ROW), (_B, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            SubtileBuffer(B, os_, 1, oi, 0),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': None,
        'rs_final': None,
    },
    # R11: outer=AG_B_COL, inner=RS_C_COL (reverse of Alg 11)
    ((_B, _COL), (_C, _COL)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A,
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B_outer, 1, sz, 0, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(get_subtile_shape(C, 1, os_), dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            set_subtile(C, result + get_subtile(C, 1, os_, 0, oi), 1, os_, 0, oi),
    },
    # R12: outer=RS_C_ROW, inner=AG_B_COL (reverse of Alg 12)
    ((_C, _ROW), (_B, _COL)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(np.copy(B), make_contiguous=False),
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, os_, 1, oi, 0),
            B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, 1, sz, 0, idx),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # R13: outer=AG_B_ROW, inner=RS_C_COL (reverse of Alg 13)
    ((_B, _ROW), (_C, _COL)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, 1, os_, 0, oi),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B_outer, 1, sz, 0, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            np.copyto(C, result + C),
    },
    # R14: outer=AG_B_ROW, inner=RS_C_ROW (reverse of Alg 14)
    ((_B, _ROW), (_C, _ROW)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A, sz, os_, idx, oi),
            B_curr=lambda A, B, C, buf, idx, sz: B_outer,
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            np.copyto(C, result + C),
    },
    # R4: outer=AG_A_COL, inner=RS_C_COL (reverse of Alg 4)
    ((_A, _COL), (_C, _COL)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: A_outer,
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, os_, sz, oi, idx),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            np.copyto(C, result + C),
    },
    # R5: outer=AG_A_COL, inner=RS_C_ROW (reverse of Alg 5)
    ((_A, _COL), (_C, _ROW)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A_outer, sz, 1, idx, 0),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, os_, 1, oi, 0),
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            np.copyto(C, result + C),
    },
    # R8: outer=RS_C_COL, inner=AG_A_ROW (reverse of Alg 8)
    ((_C, _COL), (_A, _ROW)): {
        'make_buffer': lambda A, B, C, A_outer, B_outer, oi, is_, os_:
            DoubleBuffer(A, make_contiguous=True),
        'persistent_buffer': True,
        'loopback': True,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
            B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, 1, os_, 0, oi),
            C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, sz, 1, idx, 0),
        ),
        'set_c': lambda A_outer, B_outer, oi, os_:
            lambda C, Ct, idx, sz: set_subtile(C, Ct, sz, 1, idx, 0),
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(C.shape, dtype=MATRIX_DTYPE),
        'rs_final': None,
    },
    # R9: outer=AG_A_ROW, inner=RS_C_ROW (reverse of Alg 9)
    ((_A, _ROW), (_C, _ROW)): {
        'make_buffer': None,
        'persistent_buffer': False,
        'loopback': False,
        'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
            A_curr=lambda A, B, C, buf, idx, sz: get_subtile(A_outer, sz, 1, idx, 0),
            B_curr=lambda A, B, C, buf, idx, sz: B,
            C_curr=lambda A, B, C, buf, idx, sz: np.zeros(C.shape, dtype=MATRIX_DTYPE),
        ),
        'set_c': None,
        'make_C_inner': lambda C, oi, is_, os_:
            np.zeros(get_subtile_shape(C, os_, 1), dtype=MATRIX_DTYPE),
        'rs_final': lambda result, C, oi, is_, os_:
            set_subtile(C, result + get_subtile(C, os_, 1, oi, 0), os_, 1, oi, 0),
    },
}


def _make_compute_fn(alg_key, A, B, C, inner_comm, inner_size, inner_rank,
                     outer_size, px, py, inner_config, outer_config):
    config = GEMM_2D_INNER_CONFIGS[alg_key]
    outer_is_ag = outer_config.matrix_communicated != MatrixCommunicated.C

    inner_gemm1d = Gemm1D(inner_config.matrix_communicated,
                          inner_config.subtile_scheme,
                          inner_config.communication_direction)

    # For persistent buffers, create once in factory scope
    persistent_buf = None
    if config.get('persistent_buffer') and config['make_buffer']:
        persistent_buf = config['make_buffer'](A, B, C, None, None, None, inner_size, outer_size)

    def compute_fn(A_outer, B_outer, i, outer_index):
        is_last_outer = (i == outer_size - 1)
        loopback = config['loopback'] and not is_last_outer

        # Create or reuse buffer
        if persistent_buf is not None:
            buffer = persistent_buf
        elif config['make_buffer']:
            buffer = config['make_buffer'](A, B, C, A_outer, B_outer,
                                           outer_index, inner_size, outer_size)
        else:
            buffer = None

        # Create tiles and set_c with outer context
        tiles = config['tiles'](A_outer, B_outer, outer_index, outer_size)
        set_c_fn = config['set_c'](A_outer, B_outer, outer_index, outer_size) if config['set_c'] else None

        # Determine C for inner run
        if config['make_C_inner']:
            C_inner = config['make_C_inner'](C, outer_index, inner_size, outer_size)
        else:
            C_inner = C

        # Reuse Gemm1D.run() for the inner loop
        result = inner_gemm1d.run(
            A, B, C_inner, inner_comm, inner_size, inner_rank,
            current_tiles_override=tiles,
            set_c_override=set_c_fn,
            buffer_override=buffer,
            loopback=loopback,
        )

        # Handle result based on outer type
        if config['make_C_inner']:
            if outer_is_ag:
                config['rs_final'](result, C, outer_index, inner_size, outer_size)
                return np.zeros(C.shape, dtype=MATRIX_DTYPE)
            else:
                return result
        else:
            return np.zeros(C.shape, dtype=MATRIX_DTYPE)

    return compute_fn

# Algorithm configuration dictionary
# Keyed by ordered tuples ((outer_matrix, outer_subtile), (inner_matrix, inner_subtile))
# First element = outer loop type, second = inner loop type.

GEMM_2D_ALGORITHMS = {
    # Alg 1: outer=AG_A_COL, inner=AG_A_ROW
    ((_A, _COL), (_A, _ROW)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _COL), (_A, _ROW)), *args),
    },
    # Alg 2: outer=AG_A_COL, inner=AG_B_COL
    ((_A, _COL), (_B, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _COL), (_B, _COL)), *args),
    },
    # Alg 3: outer=AG_B_ROW, inner=AG_A_COL
    ((_B, _ROW), (_A, _COL)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, px, py, oc.Get_rank(), ic.Get_rank()),
        'B_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _ROW), (_A, _COL)), *args),
    },
    # Alg 4: outer=RS_C_COL, inner=AG_A_COL
    ((_C, _COL), (_A, _COL)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m >= 1, f"m={m} must be >= 1"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: alternating_column_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _COL), (_A, _COL)), *args),
    },
    # Alg 5: outer=RS_C_ROW, inner=AG_A_COL
    ((_C, _ROW), (_A, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: alternating_column_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _ROW), (_A, _COL)), *args),
    },
    # Alg 6: outer=AG_A_ROW, inner=AG_B_COL
    ((_A, _ROW), (_B, _COL)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k >= 1, f"k={k} must be >= 1"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: alternating_row_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _ROW), (_B, _COL)), *args),
    },
    # Alg 7: outer=AG_B_ROW, inner=AG_A_ROW
    ((_B, _ROW), (_A, _ROW)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _ROW), (_A, _ROW)), *args),
    },
    # Alg 8: outer=AG_A_ROW, inner=RS_C_COL
    ((_A, _ROW), (_C, _COL)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _ROW), (_C, _COL)), *args),
    },
    # Alg 9: outer=RS_C_ROW, inner=AG_A_ROW
    ((_C, _ROW), (_A, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: A9_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _ROW), (_A, _ROW)), *args),
    },
    # Alg 10: outer=AG_B_COL, inner=AG_B_ROW
    ((_B, _COL), (_B, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _COL), (_B, _ROW)), *args),
    },
    # Alg 11: outer=RS_C_COL, inner=AG_B_COL
    ((_C, _COL), (_B, _COL)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, px, py, ic.Get_rank(), oc.Get_rank()),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: C11_get_local_indices(C, px, py, ic.Get_rank(), oc.Get_rank()),
        'flatten_gather': True,
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _COL), (_B, _COL)), *args),
    },
    # Alg 12: outer=AG_B_COL, inner=RS_C_ROW
    ((_B, _COL), (_C, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _COL), (_C, _ROW)), *args),
    },
    # Alg 13: outer=RS_C_COL, inner=AG_B_ROW
    ((_C, _COL), (_B, _ROW)): {
        'group_param_is_py': True,
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, px, py, ic.Get_rank(), oc.Get_rank()),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _COL), (_B, _ROW)), *args),
    },
    # Alg 14: outer=RS_C_ROW, inner=AG_B_ROW
    ((_C, _ROW), (_B, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n >= 1, f"n={n} must be >= 1"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, py, px, ic.Get_rank(), oc.Get_rank()),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _ROW), (_B, _ROW)), *args),
    },
    # Alg 15: outer=RS_C_COL, inner=RS_C_ROW
    ((_C, _COL), (_C, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _COL), (_C, _ROW)), *args),
    },

    # Reversed algorithm entries
    # R1: outer=AG_A_ROW, inner=AG_A_COL (reverse of Alg 1)
    ((_A, _ROW), (_A, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _ROW), (_A, _COL)), *args),
    },
    # R10: outer=AG_B_ROW, inner=AG_B_COL (reverse of Alg 10)
    ((_B, _ROW), (_B, _COL)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % py == 0, f"n={n} not divisible by py={py}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _ROW), (_B, _COL)), *args),
    },
    # R6: outer=AG_B_COL, inner=AG_A_ROW (reverse of Alg 6)
    ((_B, _COL), (_A, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k >= 1, f"k={k} must be >= 1"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: alternating_column_distribution(M, py, px, rank).copy(),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _COL), (_A, _ROW)), *args),
    },
    # R15: outer=RS_C_ROW, inner=RS_C_COL (reverse of Alg 15)
    ((_C, _ROW), (_C, _COL)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _ROW), (_C, _COL)), *args),
    },

    # R3: outer=AG_A_COL, inner=AG_B_ROW (reverse of Alg 3)
    ((_A, _COL), (_B, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _COL), (_B, _ROW)), *args),
    },
    # R2: outer=AG_B_COL, inner=AG_A_COL (reverse of Alg 2)
    ((_B, _COL), (_A, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _COL), (_A, _COL)), *args),
    },
    # R7: outer=AG_A_ROW, inner=AG_B_ROW (reverse of Alg 7)
    ((_A, _ROW), (_B, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % px == 0, f"m={m} not divisible by px={px}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _ROW), (_B, _ROW)), *args),
    },
    # R11: outer=AG_B_COL, inner=RS_C_COL (reverse of Alg 11)
    ((_B, _COL), (_C, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, py, px, oc.Get_rank(), ic.Get_rank()),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: C11_get_local_indices(C, py, px, oc.Get_rank(), ic.Get_rank()),
        'flatten_gather': True,
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _COL), (_C, _COL)), *args),
    },
    # R12: outer=RS_C_ROW, inner=AG_B_COL (reverse of Alg 12)
    ((_C, _ROW), (_B, _COL)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, px, py, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _ROW), (_B, _COL)), *args),
    },
    # R13: outer=AG_B_ROW, inner=RS_C_COL (reverse of Alg 13)
    ((_B, _ROW), (_C, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, py, px, oc.Get_rank(), ic.Get_rank()),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, row_major_distribution_get_local_indices(px, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _ROW), (_C, _COL)), *args),
    },
    # R14: outer=AG_B_ROW, inner=RS_C_ROW (reverse of Alg 14)
    ((_B, _ROW), (_C, _ROW)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n >= 1, f"n={n} must be >= 1"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: block_cyclic_distribution(M, px, py, oc.Get_rank(), ic.Get_rank()),
        'B_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_row_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_row_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_B, _ROW), (_C, _ROW)), *args),
    },
    # R4: outer=AG_A_COL, inner=RS_C_COL (reverse of Alg 4)
    ((_A, _COL), (_C, _COL)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m >= 1, f"m={m} must be >= 1"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: alternating_column_distribution(M, py, px, rank).copy(),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _COL), (_C, _COL)), *args),
    },
    # R5: outer=AG_A_COL, inner=RS_C_ROW (reverse of Alg 5)
    ((_A, _COL), (_C, _ROW)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % size == 0, f"k={k} not divisible by size={size}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: alternating_column_distribution(M, px, py, rank).copy(),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _COL), (_C, _ROW)), *args),
    },
    # R8: outer=RS_C_COL, inner=AG_A_ROW (reverse of Alg 8)
    ((_C, _COL), (_A, _ROW)): {
        'group_param_is_py': False,  # px
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % py == 0, f"m={m} not divisible by py={py}"),
            _assert(k % px == 0, f"k={k} not divisible by px={px}"),
            _assert(n % size == 0, f"n={n} not divisible by size={size}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: row_major_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, px, py, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: pure_column_distribution(M, size, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, pure_column_distribution_get_local_indices(rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_C, _COL), (_A, _ROW)), *args),
    },
    # R9: outer=AG_A_ROW, inner=RS_C_ROW (reverse of Alg 9)
    ((_A, _ROW), (_C, _ROW)): {
        'group_param_is_py': True,  # py
        'assert_div': lambda m, k, n, px, py, size: (
            _assert(m % size == 0, f"m={m} not divisible by size={size}"),
            _assert(k % py == 0, f"k={k} not divisible by py={py}"),
            _assert(n % px == 0, f"n={n} not divisible by px={px}"),
        ),
        'A_dist': lambda M, px, py, rank, size, oc, ic: A9_distribution(M, py, px, rank),
        'B_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'C_dist': lambda M, px, py, rank, size, oc, ic: col_major_distribution(M, py, px, rank),
        'get_local_indices': lambda C, rank, px, py, size, oc, ic: (C, col_major_distribution_get_local_indices(py, rank)),
        'make_compute_fn': lambda *args: _make_compute_fn(((_A, _ROW), (_C, _ROW)), *args),
    },
}


def _assert(condition, message):
    assert condition, message


class Gemm2D:
    def __init__(self, first, second):
        """Create a 2D GEMM from two 1D patterns.

        Args:
            first: Gemm1D instance for the first 1D pattern
            second: Gemm1D instance for the second 1D pattern

        Example:
            ag_a_col = Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
            ag_b_row = Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)
            gemm2d = Gemm2D(ag_a_col, ag_b_row)
            output = gemm2d.setup_and_run(m, k, n, px, py)
        """
        t1 = (first.config.matrix_communicated, first.config.subtile_scheme)
        t2 = (second.config.matrix_communicated, second.config.subtile_scheme)
        key = (t1, t2)  # ordered: first=outer, second=inner

        if key not in GEMM_2D_ALGORITHMS:
            raise ValueError(f"No 2D algorithm for combination: outer={t1}, inner={t2}")

        self.alg_config = GEMM_2D_ALGORITHMS[key]
        self.outer_gemm1d = first   # first arg is always outer
        self.inner_gemm1d = second  # second arg is always inner

    @property
    def _outer_is_ag(self):
        return self.outer_gemm1d.config.matrix_communicated != MatrixCommunicated.C

    def setup_and_run(self, m, k, n, px, py):
        comm, size, rank = mpi_setup()
        config = self.alg_config

        config['assert_div'](m, k, n, px, py, size)

        A, B, C = generate_matrices(m, k, n)
        expected = np.matmul(A, B) + C

        # Create communicators: AG outer -> remainder, RS outer -> nearby
        group_param = py if config['group_param_is_py'] else px
        if self._outer_is_ag:
            outer_comm = remainder_communicator(comm, group_param, rank)
            inner_comm = nearby_rank_communicator(comm, group_param, rank)
        else:
            outer_comm = nearby_rank_communicator(comm, group_param, rank)
            inner_comm = remainder_communicator(comm, group_param, rank)

        # Distribute matrices
        A_local = config['A_dist'](A, px, py, rank, size, outer_comm, inner_comm)
        B_local = config['B_dist'](B, px, py, rank, size, outer_comm, inner_comm)
        C_local = config['C_dist'](C, px, py, rank, size, outer_comm, inner_comm)

        C_local, elapsed_time = call_algorithm(
            self.run, comm, A_local, B_local, C_local,
            outer_comm, inner_comm, px, py
        )

        # Gather results
        gather_result = config['get_local_indices'](C_local, rank, px, py, size, outer_comm, inner_comm)
        if config.get('flatten_gather', False):
            actual_tiles_raw = comm.allgather(gather_result)
            actual_tiles = [item for sublist in actual_tiles_raw for item in sublist]
        else:
            actual_tiles = comm.allgather(gather_result)

        actual = assemble_matrix_from_tiles(actual_tiles)
        correct = matrices_equal(expected, actual)

        return create_algorithm_output(elapsed_time, correct, A, B, C, expected, actual)

    def run(self, A, B, C, outer_comm, inner_comm, px, py):
        outer_size = outer_comm.Get_size()
        outer_rank = outer_comm.Get_rank()
        inner_size = inner_comm.Get_size()
        inner_rank = inner_comm.Get_rank()

        compute_fn = self.alg_config['make_compute_fn'](
            A, B, C, inner_comm, inner_size, inner_rank,
            outer_size, px, py,
            self.inner_gemm1d.config,
            self.outer_gemm1d.config,
        )

        outer_mc = self.outer_gemm1d.config.matrix_communicated
        outer_ct = GEMM_2D_OUTER_CURRENT_TILES[outer_mc]

        return self.outer_gemm1d.run(
            A, B, C, outer_comm, outer_size, outer_rank,
            compute_fn=compute_fn,
            current_tiles_override=outer_ct,
            set_c_override=_noop_set_c,
        )


if __name__ == "__main__":
    # mpirun --oversubscribe -n 6 python composed_gemm.py
    # 1D examples:
    # gemm = Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)
    # output = gemm.setup_and_run(6,3,3,3,1)

    # 2D example:
    from debug import rank_print as rp
    algorithms = [
        ("Alg1: AG_A_COL+AG_A_ROW", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg2: AG_A_COL+AG_B_COL", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg3: AG_A_COL+AG_B_ROW", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg4: AG_A_COL+RS_C_COL", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg5: AG_A_COL+RS_C_ROW", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg6: AG_A_ROW+AG_B_COL", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg7: AG_A_ROW+AG_B_ROW", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg8: AG_A_ROW+RS_C_COL", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg9: AG_A_ROW+RS_C_ROW", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg10: AG_B_COL+AG_B_ROW", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg11: AG_B_COL+RS_C_COL", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg12: AG_B_COL+RS_C_ROW", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg13: AG_B_ROW+RS_C_COL", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV))),
        ("Alg14: AG_B_ROW+RS_C_ROW", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
        ("Alg15: RS_C_COL+RS_C_ROW", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV))),
    ]
    for name, gemm2d in algorithms:
        try:
            output = gemm2d.setup_and_run(6, 6, 6, 2, 3)
            if output['correct']:
                rp(f"{name}: PASS")
            else:
                rp(f"{name}: FAIL\nExpected:\n{output.get('expected')}\nActual:\n{output.get('actual')}")
        except Exception as e:
            rp(f"{name}: ERROR - {e}")


