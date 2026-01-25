from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from constants import MATRIX_DTYPE
from data_classes import CurrentTiles, DistributionFunctions, DivisibiltyRequirements
from debug import print_full_matrices, rank_print
from distribution import get_subtile, pure_column_distribution, pure_column_distribution_get_local_indices, pure_row_distribution, pure_row_distribution_get_local_indices, set_subtile
from enums import CommunicationDirection, ConfigurationOptions1D, GemmDimension, MatrixCommunicated, SubtileScheme
from util import AccumulationBuffer, DoubleBuffer, assemble_matrix_from_tiles, call_algorithm, create_algorithm_output, generate_matrices, matrices_equal, mpi_setup, receive, send

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

    def run(self, A, B, C, comm, size, rank):
        index = self.config.index(rank, size, self.config.direction_increment)
        loop_iterations = size
        buffer = self.config.buffer(A, B, C)

        for i in range(loop_iterations):
            if self.config.matrix_communicated != MatrixCommunicated.C:
                if i != loop_iterations - 1:
                    send_request, receive_request = send(comm, buffer.get_send_tile(), buffer.get_receive_tile(), direction=self.config.communication_direction)
            
            A_curr = self.config.current_tiles.A_curr(A, B, C, buffer, index, size)
            B_curr = self.config.current_tiles.B_curr(A, B, C, buffer, index, size)

            local_result = np.matmul(A_curr, B_curr)

            if self.config.matrix_communicated != MatrixCommunicated.C or i == 0:
                C_curr = self.config.current_tiles.C_curr(A, B, C, buffer, index, size)
            else:
                C_curr = receive([receive_request, send_request], buffer.on_receive())

            C_tmp = local_result + C_curr

            if self.config.matrix_communicated != MatrixCommunicated.C:
                self.config.set_c(C, C_tmp, index, size)

                if i != loop_iterations - 1:
                    receive([send_request, receive_request], buffer.on_receive())
            else:
                if i == loop_iterations - 1:
                    C = C + C_tmp
                else:
                    send_request, receive_request = send(comm, buffer.get_send_tile(C_tmp), buffer.get_receive_tile(), direction=self.config.communication_direction)

            index = (index + self.config.direction_increment) % size

        return C



if __name__ == "__main__":
    # mpirun --oversubscribe -n 3 python composed_gemm.py
    # gemm = Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)
    # gemm = Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)
    # gemm = Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)
    # gemm = Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)
    # gemm = Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)
    gemm = Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)
    output = gemm.setup_and_run(6,3,3,3,1)
    output_string = f"Runtime: {output.get('elapsed_time')}, Correct: {output.get('correct')}\nExpected:\n{output.get('expected')}\nActual:\n{output.get('actual')}"
    rank_print(output_string)


# 2d plan
# AG_A_COL_AG_B_ROW example for how I want the interface to look
# ag_a_col = Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV) # could also be SEND_NEXT shouldn't matter
# ag_b_row = Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)
# gemm2d = Gemm2D(outer=ag_a_col, inner=ag_b_row)
# output = gemm2d.setup_and_run(...)