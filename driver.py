from mpi4py import MPI
import numpy as np
import argparse

from debug import rank_print, parallel_print

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


gemm_algorithms = {
    "AG_A_COL_AG_A_ROW": AG_A_COL_AG_A_ROW,
    "AG_A_COL_AG_B_COL": AG_A_COL_AG_B_COL,
    "AG_A_COL_AG_B_ROW": AG_A_COL_AG_B_ROW,
    "AG_A_COL_RS_C_COL": AG_A_COL_RS_C_COL,
    "AG_A_COL_RS_C_ROW": AG_A_COL_RS_C_ROW,
    "AG_A_ROW_AG_B_COL": AG_A_ROW_AG_B_COL,
    "AG_A_ROW_AG_B_ROW": AG_A_ROW_AG_B_ROW,
    "AG_A_ROW_RS_C_COL": AG_A_ROW_RS_C_COL,
    "AG_A_ROW_RS_C_ROW": AG_A_ROW_RS_C_ROW,
    "AG_B_COL_AG_B_ROW": AG_B_COL_AG_B_ROW,
    "AG_B_COL_RS_C_COL": AG_B_COL_RS_C_COL,
    "AG_B_COL_RS_C_ROW": AG_B_COL_RS_C_ROW,
    "AG_B_ROW_RS_C_COL": AG_B_ROW_RS_C_COL,
    "AG_B_ROW_RS_C_ROW": AG_B_ROW_RS_C_ROW,
    "RS_C_COL_RS_C_ROW": RS_C_COL_RS_C_ROW,

    "AG_A_COL": AG_A_COL,
    "AG_A_ROW": AG_A_ROW,
    "AG_B_COL": AG_B_COL,
    "AG_B_ROW": AG_B_ROW,
    "RS_C_COL": RS_C_COL,
    "RS_C_ROW": RS_C_ROW
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a GEMM algorithm with specified parameters.")
    parser.add_argument(
        "-a", "--algorithm", type=str, required=True,
        help="The name of the GEMM algorithm to run."
    )
    parser.add_argument(
        "-m", type=int, required=True,
        help="The value for M dimension."
    )
    parser.add_argument(
        "-k", type=int, required=True,
        help="The value for K dimension."
    )
    parser.add_argument(
        "-n", type=int, required=True,
        help="The value for N dimension."
    )
    parser.add_argument(
        "-px", type=int, required=True,
        help="Number of Processors in the First Dimension."
    )
    parser.add_argument(
        "-py", type=int, required=True,
        help="Number of Processors in the Second Dimension."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    algorithm = gemm_algorithms.get(args.algorithm)
    if algorithm is None:
        rank_print("INVALID ALGORITHM PROVIDED")
        return
    
    m, k, n = args.m, args.k, args.n
    px, py = args.px, args.py

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if px * py != size:
        rank_print(f"INVALID PX AND PY DISTRIBUTION: {px} * {py} != {size}")
        return

    rank_print(f"Algorithm: {algorithm.__name__}, M: {m}, K: {k}, N: {n}")
    output = algorithm(m, k, n, px, py)
    output_string = f"Runtime: {output.get('elapsed_time')}, Correct: {output.get('correct')}\nExpected:\n{output.get('expected')}\nActual:\n{output.get('actual')}"
    rank_print(output_string)


if __name__ == "__main__":
    main()
