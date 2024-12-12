from enum import Enum, auto
from math import sqrt
import random
from mpi4py import MPI

from debug import rank_print
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
    RS_C_COL_RS_C_ROW
)

class MinGemmDimension(Enum):
    PX = auto()
    PY = auto()
    SIZE = auto()

class TestGemmConfiguration():
    def __init__(self, algorithm, min_m, min_k, min_n):
        # min dims are elements of MinGemmDimension Enum
        self.algorithm = algorithm
        self.min_m = min_m
        self.min_k = min_k
        self.min_n = min_n

    def get_min_dimensions(self, size, px, py):
        min_m = px if self.min_m == MinGemmDimension.PX else py if self.min_m == MinGemmDimension.PY else size
        min_k = size if self.min_k == MinGemmDimension.SIZE else px if self.min_k == MinGemmDimension.PX else py
        min_n = py if self.min_n == MinGemmDimension.PY else size if self.min_n == MinGemmDimension.SIZE else px

        return min_m, min_k, min_n


class TestGemm():

    ITERATIONS = 50
    MULTIPLIER_RANGE = 50

    def get_factors(self, num):
        factors = []
        for i in range(1, int(sqrt(num)) + 1):
            if num % i == 0:
                factors.append(i)
            if i != num // i:
                factors.append(num // i)
        return factors

    def test_2d_gemm(self, config):
        # 3
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

            output = config.algorithm(m, k, n, px, py)
            comm.Barrier()

            error_string = f"{config.algorithm.__name__} FAILED\nParameters: m={m}, k={k}, n={n}, px={px}, py={py}\nExpected: {output.get('expected')}\nActual: {output.get('actual')}"

            if rank == 0:
                assert output.get("correct"), error_string

        rank_print(f"{config.algorithm.__name__} TESTS PASS")
        


GEMM_TESTING_CONFIGURATIONS = {
    "AG_A_COL_AG_B_COL": TestGemmConfiguration(AG_A_COL_AG_B_COL, MinGemmDimension.PX, MinGemmDimension.PY, MinGemmDimension.SIZE),
    "AG_A_COL_AG_B_ROW": TestGemmConfiguration(AG_A_COL_AG_B_ROW, MinGemmDimension.PX, MinGemmDimension.SIZE, MinGemmDimension.PY)
}

def main():
    tests = TestGemm()
    for config in GEMM_TESTING_CONFIGURATIONS.values():
        tests.test_2d_gemm(config)


if __name__ == "__main__":
    main()
