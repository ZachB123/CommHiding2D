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
    ONE = auto()
    PX = auto()
    PY = auto()
    SIZE = auto()

class TestGemmConfiguration:
    def __init__(self, algorithm, min_m, min_k, min_n):
        # min dims are elements of MinGemmDimension Enum
        self.algorithm = algorithm
        self.min_m = min_m
        self.min_k = min_k
        self.min_n = min_n

    def _resolve_dim(self, dim, size, px, py):
        if dim == MinGemmDimension.PX:
            return px
        elif dim == MinGemmDimension.PY:
            return py
        elif dim == MinGemmDimension.SIZE:
            return size
        elif dim == MinGemmDimension.ONE:
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
    "AG_A_COL_AG_A_ROW": TestGemmConfiguration(AG_A_COL_AG_A_ROW, MinGemmDimension.PX, MinGemmDimension.PY, MinGemmDimension.SIZE), # 1
    "AG_A_COL_AG_B_COL": TestGemmConfiguration(AG_A_COL_AG_B_COL, MinGemmDimension.PX, MinGemmDimension.PY, MinGemmDimension.SIZE), # 2
    "AG_A_COL_AG_B_ROW": TestGemmConfiguration(AG_A_COL_AG_B_ROW, MinGemmDimension.PX, MinGemmDimension.SIZE, MinGemmDimension.PY), # 3
    "AG_A_COL_RS_C_COL": TestGemmConfiguration(AG_A_COL_RS_C_COL, MinGemmDimension.ONE, MinGemmDimension.SIZE, MinGemmDimension.SIZE), # 4
    "AG_A_COL_RS_C_ROW": TestGemmConfiguration(AG_A_COL_RS_C_ROW, MinGemmDimension.PX, MinGemmDimension.SIZE, MinGemmDimension.PY), # 5
    "AG_A_ROW_AG_B_COL": TestGemmConfiguration(AG_A_ROW_AG_B_COL, MinGemmDimension.SIZE, MinGemmDimension.ONE, MinGemmDimension.SIZE), # 6
    "AG_A_ROW_AG_B_ROW": TestGemmConfiguration(AG_A_ROW_AG_B_ROW, MinGemmDimension.SIZE, MinGemmDimension.PX, MinGemmDimension.PY), # 7
    "AG_A_ROW_RS_C_COL": TestGemmConfiguration(AG_A_ROW_RS_C_COL, MinGemmDimension.PX, MinGemmDimension.PY, MinGemmDimension.SIZE), # 8

    "AG_B_COL_AG_B_ROW": TestGemmConfiguration(AG_B_COL_AG_B_ROW, MinGemmDimension.SIZE, MinGemmDimension.PX, MinGemmDimension.PY), # 10
}

def main():
    tests = TestGemm()
    for config in GEMM_TESTING_CONFIGURATIONS.values():
        tests.test_2d_gemm(config)


if __name__ == "__main__":
    main()
