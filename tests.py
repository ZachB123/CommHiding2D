from enum import Enum, auto
import random
import argparse
from mpi4py import MPI

from composed_gemm import CommunicationDirection, Gemm1D, Gemm2D, GemmDimension, MatrixCommunicated, SubtileScheme
from constants import USE_REFACTORED_ALGORITHMS
from debug import rank_print

if USE_REFACTORED_ALGORITHMS:
    from refactored_gemm import (
        AG_A_COL_AG_A_ROW,
        AG_A_COL_AG_B_COL,
        AG_A_COL_AG_B_ROW,
        AG_A_COL_RS_C_COL,
        AG_A_COL_RS_C_ROW,
        AG_A_ROW_AG_B_COL,
        AG_A_ROW_AG_B_ROW,
        AG_A_ROW_RS_C_COL,
        AG_A_ROW_RS_C_ROW,
        AG_B_COL_AG_B_ROW,
        AG_B_COL_RS_C_COL,
        AG_B_COL_RS_C_ROW,
        AG_B_ROW_RS_C_COL,
        AG_B_ROW_RS_C_ROW,
        RS_C_COL_RS_C_ROW,

        AG_A_COL,
        AG_A_ROW,
        AG_B_COL,
        AG_B_ROW,
        RS_C_COL,
        RS_C_ROW
    )
else:
    from gemm import (
        AG_A_COL_AG_A_ROW,
        AG_A_COL_AG_B_COL,
        AG_A_COL_AG_B_ROW,
        AG_A_COL_RS_C_COL,
        AG_A_COL_RS_C_ROW,
        AG_A_ROW_AG_B_COL,
        AG_A_ROW_AG_B_ROW,
        AG_A_ROW_RS_C_COL,
        AG_A_ROW_RS_C_ROW,
        AG_B_COL_AG_B_ROW,
        AG_B_COL_RS_C_COL,
        AG_B_COL_RS_C_ROW,
        AG_B_ROW_RS_C_COL,
        AG_B_ROW_RS_C_ROW,
        RS_C_COL_RS_C_ROW,

        AG_A_COL,
        AG_A_ROW,
        AG_B_COL,
        AG_B_ROW,
        RS_C_COL,
        RS_C_ROW
    )

class TestGemmConfiguration:
    def __init__(self, algorithm, min_m, min_k, min_n):
        # min dims are elements of MinGemmDimension Enum
        self.algorithm = algorithm
        self.min_m = min_m
        self.min_k = min_k
        self.min_n = min_n

    def _resolve_dim(self, dim, size, px, py):
        if dim == GemmDimension.PX:
            return px
        elif dim == GemmDimension.PY:
            return py
        elif dim == GemmDimension.SIZE:
            return size
        elif dim == GemmDimension.ONE:
            return 1
        else:
            raise ValueError(f"Unknown MinGemmDimension: {dim}")

    def get_min_dimensions(self, size, px, py):
        min_m = self._resolve_dim(self.min_m, size, px, py)
        min_k = self._resolve_dim(self.min_k, size, px, py)
        min_n = self._resolve_dim(self.min_n, size, px, py)
        return min_m, min_k, min_n


class TestGemm():

    ITERATIONS = 30
    MULTIPLIER_RANGE = 50

    def get_factors(self, n):
        if n <= 0:
            raise ValueError("Input must be a positive integer.")
        
        factors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        
        return sorted(factors)
    

    def test_2d_gemm(self, config):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        factors = self.get_factors(size)    
        random.seed(42)

        for i in range(TestGemm.ITERATIONS):

            rank_print(f"{i + 1}/{TestGemm.ITERATIONS}", flush=True)

            px = random.sample(factors, 1)[0]
            py = size // px
            
            min_m, min_k, min_n = config.get_min_dimensions(size, px, py)
            
            m = min_m * random.randint(1, TestGemm.MULTIPLIER_RANGE)
            k = min_k * random.randint(1, TestGemm.MULTIPLIER_RANGE)
            n = min_n * random.randint(1, TestGemm.MULTIPLIER_RANGE)

            try:
                # comm.Barrier()
                output = config.algorithm(m, k, n, px, py)
                # comm.Barrier()
            except Exception as e:
                error_message = (
                    f"Exception occurred in {config.algorithm.__name__}\n"
                    f"Parameters: m={m}, k={k}, n={n}, px={px}, py={py}\n"
                    f"Error: {e}"
                )
                print(error_message)
                raise e

            error_string = f"{config.algorithm.__name__} FAILED\nParameters: m={m}, k={k}, n={n}, px={px}, py={py}\nExpected: {output.get('expected')}\nActual: {output.get('actual')}"

            if rank == 0:
                assert output.get("correct"), error_string

        rank_print(f"{config.algorithm.__name__} TESTS PASS")
        


GEMM_TESTING_CONFIGURATIONS = {
    "AG_A_COL_AG_A_ROW": TestGemmConfiguration(AG_A_COL_AG_A_ROW, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE), # 1
    "AG_A_COL_AG_B_COL": TestGemmConfiguration(AG_A_COL_AG_B_COL, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE), # 2
    "AG_A_COL_AG_B_ROW": TestGemmConfiguration(AG_A_COL_AG_B_ROW, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY), # 3
    "AG_A_COL_RS_C_COL": TestGemmConfiguration(AG_A_COL_RS_C_COL, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE), # 4
    "AG_A_COL_RS_C_ROW": TestGemmConfiguration(AG_A_COL_RS_C_ROW, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY), # 5
    "AG_A_ROW_AG_B_COL": TestGemmConfiguration(AG_A_ROW_AG_B_COL, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE), # 6
    "AG_A_ROW_AG_B_ROW": TestGemmConfiguration(AG_A_ROW_AG_B_ROW, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY), # 7
    "AG_A_ROW_RS_C_COL": TestGemmConfiguration(AG_A_ROW_RS_C_COL, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE), # 8
    "AG_A_ROW_RS_C_ROW": TestGemmConfiguration(AG_A_ROW_RS_C_ROW, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY), # 9
    "AG_B_COL_AG_B_ROW": TestGemmConfiguration(AG_B_COL_AG_B_ROW, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY), # 10
    "AG_B_COL_RS_C_COL": TestGemmConfiguration(AG_B_COL_RS_C_COL, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE), # 11
    "AG_B_COL_RS_C_ROW": TestGemmConfiguration(AG_B_COL_RS_C_ROW, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY), # 12
    "AG_B_ROW_RS_C_COL": TestGemmConfiguration(AG_B_ROW_RS_C_COL, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY), # 13
    "AG_B_ROW_RS_C_ROW": TestGemmConfiguration(AG_B_ROW_RS_C_ROW, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE), # 14
    "RS_C_COL_RS_C_ROW": TestGemmConfiguration(RS_C_COL_RS_C_ROW, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY), # 15

    "AG_A_COL": TestGemmConfiguration(AG_A_COL, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_ROW": TestGemmConfiguration(AG_A_ROW, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL": TestGemmConfiguration(AG_B_COL, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW": TestGemmConfiguration(AG_B_ROW, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL": TestGemmConfiguration(RS_C_COL, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW": TestGemmConfiguration(RS_C_ROW, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),

    "AG_A_COL_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_ROW_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW_PARAMETERIZED_PREV": TestGemmConfiguration(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),

    "AG_A_COL_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_ROW_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW_PARAMETERIZED_NEXT": TestGemmConfiguration(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),

    # Composed 2D algorithms — format: OUTER_MATRIX_DIST_DIR_INNER_MATRIX_DIST_DIR_COMPOSED
    # First Gemm1D = outer loop, second Gemm1D = inner loop
    # Alg 1: outer=AG_A_COL, inner=AG_A_ROW
    "AG_A_COL_PREV_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # Alg 2: outer=AG_A_COL, inner=AG_B_COL
    "AG_A_COL_PREV_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # Alg 3: outer=AG_B_ROW, inner=AG_A_COL
    "AG_B_ROW_PREV_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    # Alg 4: outer=RS_C_COL, inner=AG_A_COL
    "RS_C_COL_PREV_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    # Alg 5: outer=RS_C_ROW, inner=AG_A_COL
    "RS_C_ROW_PREV_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    # Alg 6: outer=AG_A_ROW, inner=AG_B_COL
    "AG_A_ROW_PREV_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    # Alg 7: outer=AG_B_ROW, inner=AG_A_ROW
    "AG_B_ROW_PREV_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    # Alg 8: outer=AG_A_ROW, inner=RS_C_COL
    "AG_A_ROW_PREV_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # Alg 9: outer=RS_C_ROW, inner=AG_A_ROW
    "RS_C_ROW_PREV_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    # Alg 10: outer=AG_B_COL, inner=AG_B_ROW
    "AG_B_COL_PREV_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    # Alg 11: outer=RS_C_COL, inner=AG_B_COL
    "RS_C_COL_PREV_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # Alg 12: outer=AG_B_COL, inner=RS_C_ROW
    "AG_B_COL_PREV_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    # Alg 13: outer=RS_C_COL, inner=AG_B_ROW
    "RS_C_COL_PREV_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    # Alg 14: outer=RS_C_ROW, inner=AG_B_ROW
    "RS_C_ROW_PREV_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    # Alg 15: outer=RS_C_COL, inner=RS_C_ROW
    "RS_C_COL_PREV_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),

    # === Direction variants: PREV+NEXT ===
    "AG_A_COL_PREV_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_PREV_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_ROW_PREV_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_COL_PREV_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW_PREV_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "AG_A_ROW_PREV_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW_PREV_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_A_ROW_PREV_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "RS_C_ROW_PREV_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_COL_PREV_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_PREV_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_COL_PREV_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_PREV_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_ROW_PREV_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL_PREV_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),

    # === Direction variants: NEXT+PREV ===
    "AG_A_COL_NEXT_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_NEXT_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_ROW_NEXT_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW_NEXT_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "AG_A_ROW_NEXT_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW_NEXT_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_A_ROW_NEXT_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "RS_C_ROW_NEXT_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_COL_NEXT_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_COL_NEXT_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_ROW_NEXT_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL_NEXT_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),

    # === Direction variants: NEXT+NEXT ===
    "AG_A_COL_NEXT_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_NEXT_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_ROW_NEXT_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "RS_C_ROW_NEXT_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "AG_A_ROW_NEXT_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_ROW_NEXT_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_A_ROW_NEXT_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "RS_C_ROW_NEXT_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_COL_NEXT_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_B_COL_NEXT_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "RS_C_COL_NEXT_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),
    "RS_C_ROW_NEXT_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "RS_C_COL_NEXT_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.SIZE, GemmDimension.PY),

    # === Reversed algorithm tests ===
    # R1: outer=AG_A_ROW, inner=AG_A_COL — min dims: m%py, k%px, n%size
    "AG_A_ROW_PREV_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_A_ROW_PREV_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_A_ROW_NEXT_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_A_ROW_NEXT_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    # R10: outer=AG_B_ROW, inner=AG_B_COL — min dims: m%size, k%px, n%py
    "AG_B_ROW_PREV_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_ROW_PREV_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_ROW_NEXT_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    "AG_B_ROW_NEXT_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PX, GemmDimension.PY),
    # R15: outer=RS_C_ROW, inner=RS_C_COL — min dims: m%py, k%size, n%px
    "RS_C_ROW_PREV_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "RS_C_ROW_PREV_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "RS_C_ROW_NEXT_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "RS_C_ROW_NEXT_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    # R6: outer=AG_B_COL, inner=AG_A_ROW — min dims: m%size, k%1, n%size
    "AG_B_COL_PREV_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL_PREV_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL_NEXT_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    "AG_B_COL_NEXT_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.ONE, GemmDimension.SIZE),
    # R3: outer=AG_A_COL, inner=AG_B_ROW — min dims: m%px, k%py, n%size
    "AG_A_COL_PREV_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_PREV_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_NEXT_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_COL_NEXT_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # R2: outer=AG_B_COL, inner=AG_A_COL — min dims: m%size, k%py, n%px
    "AG_B_COL_PREV_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_B_COL_PREV_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_B_COL_NEXT_AG_A_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_B_COL_NEXT_AG_A_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    # R7: outer=AG_A_ROW, inner=AG_B_ROW — min dims: m%px, k%py, n%size
    "AG_A_ROW_PREV_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_ROW_PREV_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_ROW_NEXT_AG_B_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    "AG_A_ROW_NEXT_AG_B_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PX, GemmDimension.PY, GemmDimension.SIZE),
    # R11: outer=AG_B_COL, inner=RS_C_COL — min dims: m%py, k%px, n%size
    "AG_B_COL_PREV_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_B_COL_PREV_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_B_COL_NEXT_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "AG_B_COL_NEXT_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    # R12: outer=RS_C_ROW, inner=AG_B_COL — min dims: m%size, k%py, n%px
    "RS_C_ROW_PREV_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "RS_C_ROW_PREV_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "RS_C_ROW_NEXT_AG_B_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "RS_C_ROW_NEXT_AG_B_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    # R13: outer=AG_B_ROW, inner=RS_C_COL — min dims: m%py, k%size, n%px
    "AG_B_ROW_PREV_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_B_ROW_PREV_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_B_ROW_NEXT_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_B_ROW_NEXT_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    # R14: outer=AG_B_ROW, inner=RS_C_ROW — min dims: m%size, k%size, n%1
    "AG_B_ROW_PREV_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "AG_B_ROW_PREV_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "AG_B_ROW_NEXT_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    "AG_B_ROW_NEXT_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.SIZE, GemmDimension.ONE),
    # R4: outer=AG_A_COL, inner=RS_C_COL — min dims: m%1, k%size, n%size
    "AG_A_COL_PREV_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_COL_PREV_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_COL_NEXT_RS_C_COL_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    "AG_A_COL_NEXT_RS_C_COL_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.ONE, GemmDimension.SIZE, GemmDimension.SIZE),
    # R5: outer=AG_A_COL, inner=RS_C_ROW — min dims: m%py, k%size, n%px
    "AG_A_COL_PREV_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_A_COL_PREV_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_A_COL_NEXT_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    "AG_A_COL_NEXT_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.SIZE, GemmDimension.PX),
    # R8: outer=RS_C_COL, inner=AG_A_ROW — min dims: m%py, k%px, n%size
    "RS_C_COL_PREV_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "RS_C_COL_PREV_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "RS_C_COL_NEXT_AG_A_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    "RS_C_COL_NEXT_AG_A_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.PY, GemmDimension.PX, GemmDimension.SIZE),
    # R9: outer=AG_A_ROW, inner=RS_C_ROW — min dims: m%size, k%py, n%px
    "AG_A_ROW_PREV_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_A_ROW_PREV_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_A_ROW_NEXT_RS_C_ROW_PREV_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
    "AG_A_ROW_NEXT_RS_C_ROW_NEXT_COMPOSED": TestGemmConfiguration(Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run, GemmDimension.SIZE, GemmDimension.PY, GemmDimension.PX),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test GEMM algorithms')
    parser.add_argument(
        '-a', '--algorithm',
        type=str,
        choices=list(GEMM_TESTING_CONFIGURATIONS.keys()),
        help='Specify a single algorithm to test'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    tests = TestGemm()
    
    if args.algorithm:
        config = GEMM_TESTING_CONFIGURATIONS[args.algorithm]
        tests.test_2d_gemm(config)
    else:
        # Run all algorithms
        for config in GEMM_TESTING_CONFIGURATIONS.values():
            tests.test_2d_gemm(config)


if __name__ == "__main__":
    main()









# algorithm names
# AG_A_COL_PREV_AG_A_ROW_PREV_COMPOSED
# AG_A_COL_PREV_AG_A_ROW_NEXT_COMPOSED
# AG_A_COL_NEXT_AG_A_ROW_PREV_COMPOSED
# AG_A_COL_NEXT_AG_A_ROW_NEXT_COMPOSED
# AG_A_ROW_PREV_AG_A_COL_PREV_COMPOSED
# AG_A_ROW_PREV_AG_A_COL_NEXT_COMPOSED
# AG_A_ROW_NEXT_AG_A_COL_PREV_COMPOSED
# AG_A_ROW_NEXT_AG_A_COL_NEXT_COMPOSED
# AG_A_COL_PREV_AG_B_COL_PREV_COMPOSED
# AG_A_COL_PREV_AG_B_COL_NEXT_COMPOSED
# AG_A_COL_NEXT_AG_B_COL_PREV_COMPOSED
# AG_A_COL_NEXT_AG_B_COL_NEXT_COMPOSED
# AG_B_COL_PREV_AG_A_COL_PREV_COMPOSED
# AG_B_COL_PREV_AG_A_COL_NEXT_COMPOSED
# AG_B_COL_NEXT_AG_A_COL_PREV_COMPOSED
# AG_B_COL_NEXT_AG_A_COL_NEXT_COMPOSED
# AG_A_COL_PREV_AG_B_ROW_PREV_COMPOSED
# AG_A_COL_PREV_AG_B_ROW_NEXT_COMPOSED
# AG_A_COL_NEXT_AG_B_ROW_PREV_COMPOSED
# AG_A_COL_NEXT_AG_B_ROW_NEXT_COMPOSED
# AG_B_ROW_PREV_AG_A_COL_PREV_COMPOSED
# AG_B_ROW_PREV_AG_A_COL_NEXT_COMPOSED
# AG_B_ROW_NEXT_AG_A_COL_PREV_COMPOSED
# AG_B_ROW_NEXT_AG_A_COL_NEXT_COMPOSED
# AG_A_COL_PREV_RS_C_COL_PREV_COMPOSED
# AG_A_COL_PREV_RS_C_COL_NEXT_COMPOSED
# AG_A_COL_NEXT_RS_C_COL_PREV_COMPOSED
# AG_A_COL_NEXT_RS_C_COL_NEXT_COMPOSED
# RS_C_COL_PREV_AG_A_COL_PREV_COMPOSED
# RS_C_COL_PREV_AG_A_COL_NEXT_COMPOSED
# RS_C_COL_NEXT_AG_A_COL_PREV_COMPOSED
# RS_C_COL_NEXT_AG_A_COL_NEXT_COMPOSED
# AG_A_COL_PREV_RS_C_ROW_PREV_COMPOSED
# AG_A_COL_PREV_RS_C_ROW_NEXT_COMPOSED
# AG_A_COL_NEXT_RS_C_ROW_PREV_COMPOSED
# AG_A_COL_NEXT_RS_C_ROW_NEXT_COMPOSED
# RS_C_ROW_PREV_AG_A_COL_PREV_COMPOSED
# RS_C_ROW_PREV_AG_A_COL_NEXT_COMPOSED
# RS_C_ROW_NEXT_AG_A_COL_PREV_COMPOSED
# RS_C_ROW_NEXT_AG_A_COL_NEXT_COMPOSED
# AG_A_ROW_PREV_AG_B_COL_PREV_COMPOSED
# AG_A_ROW_PREV_AG_B_COL_NEXT_COMPOSED
# AG_A_ROW_NEXT_AG_B_COL_PREV_COMPOSED
# AG_A_ROW_NEXT_AG_B_COL_NEXT_COMPOSED
# AG_B_COL_PREV_AG_A_ROW_PREV_COMPOSED
# AG_B_COL_PREV_AG_A_ROW_NEXT_COMPOSED
# AG_B_COL_NEXT_AG_A_ROW_PREV_COMPOSED
# AG_B_COL_NEXT_AG_A_ROW_NEXT_COMPOSED
# AG_A_ROW_PREV_AG_B_ROW_PREV_COMPOSED
# AG_A_ROW_PREV_AG_B_ROW_NEXT_COMPOSED
# AG_A_ROW_NEXT_AG_B_ROW_PREV_COMPOSED
# AG_A_ROW_NEXT_AG_B_ROW_NEXT_COMPOSED
# AG_B_ROW_PREV_AG_A_ROW_PREV_COMPOSED
# AG_B_ROW_PREV_AG_A_ROW_NEXT_COMPOSED
# AG_B_ROW_NEXT_AG_A_ROW_PREV_COMPOSED
# AG_B_ROW_NEXT_AG_A_ROW_NEXT_COMPOSED
# AG_A_ROW_PREV_RS_C_COL_PREV_COMPOSED
# AG_A_ROW_PREV_RS_C_COL_NEXT_COMPOSED
# AG_A_ROW_NEXT_RS_C_COL_PREV_COMPOSED
# AG_A_ROW_NEXT_RS_C_COL_NEXT_COMPOSED
# RS_C_COL_PREV_AG_A_ROW_PREV_COMPOSED
# RS_C_COL_PREV_AG_A_ROW_NEXT_COMPOSED
# RS_C_COL_NEXT_AG_A_ROW_PREV_COMPOSED
# RS_C_COL_NEXT_AG_A_ROW_NEXT_COMPOSED
# AG_A_ROW_PREV_RS_C_ROW_PREV_COMPOSED
# AG_A_ROW_PREV_RS_C_ROW_NEXT_COMPOSED
# AG_A_ROW_NEXT_RS_C_ROW_PREV_COMPOSED
# AG_A_ROW_NEXT_RS_C_ROW_NEXT_COMPOSED
# RS_C_ROW_PREV_AG_A_ROW_PREV_COMPOSED
# RS_C_ROW_PREV_AG_A_ROW_NEXT_COMPOSED
# RS_C_ROW_NEXT_AG_A_ROW_PREV_COMPOSED
# RS_C_ROW_NEXT_AG_A_ROW_NEXT_COMPOSED
# AG_B_COL_PREV_AG_B_ROW_PREV_COMPOSED
# AG_B_COL_PREV_AG_B_ROW_NEXT_COMPOSED
# AG_B_COL_NEXT_AG_B_ROW_PREV_COMPOSED
# AG_B_COL_NEXT_AG_B_ROW_NEXT_COMPOSED
# AG_B_ROW_PREV_AG_B_COL_PREV_COMPOSED
# AG_B_ROW_PREV_AG_B_COL_NEXT_COMPOSED
# AG_B_ROW_NEXT_AG_B_COL_PREV_COMPOSED
# AG_B_ROW_NEXT_AG_B_COL_NEXT_COMPOSED
# AG_B_COL_PREV_RS_C_COL_PREV_COMPOSED
# AG_B_COL_PREV_RS_C_COL_NEXT_COMPOSED
# AG_B_COL_NEXT_RS_C_COL_PREV_COMPOSED
# AG_B_COL_NEXT_RS_C_COL_NEXT_COMPOSED
# RS_C_COL_PREV_AG_B_COL_PREV_COMPOSED
# RS_C_COL_PREV_AG_B_COL_NEXT_COMPOSED
# RS_C_COL_NEXT_AG_B_COL_PREV_COMPOSED
# RS_C_COL_NEXT_AG_B_COL_NEXT_COMPOSED
# AG_B_COL_PREV_RS_C_ROW_PREV_COMPOSED
# AG_B_COL_PREV_RS_C_ROW_NEXT_COMPOSED
# AG_B_COL_NEXT_RS_C_ROW_PREV_COMPOSED
# AG_B_COL_NEXT_RS_C_ROW_NEXT_COMPOSED
# RS_C_ROW_PREV_AG_B_COL_PREV_COMPOSED
# RS_C_ROW_PREV_AG_B_COL_NEXT_COMPOSED
# RS_C_ROW_NEXT_AG_B_COL_PREV_COMPOSED
# RS_C_ROW_NEXT_AG_B_COL_NEXT_COMPOSED
# AG_B_ROW_PREV_RS_C_COL_PREV_COMPOSED
# AG_B_ROW_PREV_RS_C_COL_NEXT_COMPOSED
# AG_B_ROW_NEXT_RS_C_COL_PREV_COMPOSED
# AG_B_ROW_NEXT_RS_C_COL_NEXT_COMPOSED
# RS_C_COL_PREV_AG_B_ROW_PREV_COMPOSED
# RS_C_COL_PREV_AG_B_ROW_NEXT_COMPOSED
# RS_C_COL_NEXT_AG_B_ROW_PREV_COMPOSED
# RS_C_COL_NEXT_AG_B_ROW_NEXT_COMPOSED
# AG_B_ROW_PREV_RS_C_ROW_PREV_COMPOSED
# AG_B_ROW_PREV_RS_C_ROW_NEXT_COMPOSED
# AG_B_ROW_NEXT_RS_C_ROW_PREV_COMPOSED
# AG_B_ROW_NEXT_RS_C_ROW_NEXT_COMPOSED
# RS_C_ROW_PREV_AG_B_ROW_PREV_COMPOSED
# RS_C_ROW_PREV_AG_B_ROW_NEXT_COMPOSED
# RS_C_ROW_NEXT_AG_B_ROW_PREV_COMPOSED
# RS_C_ROW_NEXT_AG_B_ROW_NEXT_COMPOSED
# RS_C_COL_PREV_RS_C_ROW_PREV_COMPOSED
# RS_C_COL_PREV_RS_C_ROW_NEXT_COMPOSED
# RS_C_COL_NEXT_RS_C_ROW_PREV_COMPOSED
# RS_C_COL_NEXT_RS_C_ROW_NEXT_COMPOSED
# RS_C_ROW_PREV_RS_C_COL_PREV_COMPOSED
# RS_C_ROW_PREV_RS_C_COL_NEXT_COMPOSED
# RS_C_ROW_NEXT_RS_C_COL_PREV_COMPOSED
# RS_C_ROW_NEXT_RS_C_COL_NEXT_COMPOSED