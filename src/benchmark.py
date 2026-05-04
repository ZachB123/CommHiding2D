import argparse
import csv
import functools
import gc
import itertools
import os
import sys
from datetime import datetime

import numpy as np
from tqdm import tqdm

from constants import BENCHMARK_FOLDER, MATRIX_DTYPE
from data_classes import BenchmarkAlgorithm
from composed_gemm_1d import CommunicationDirection, Gemm1D, MatrixCommunicated, SubtileScheme
from util import mpi_setup
from composed_gemm_2d import Gemm2D
import gemm


SMALL_DIMENSIONS  = [1440, 2880, 4320, 7200]
MEDIUM_DIMENSIONS = [4320, 7200, 10080, 14400]
LARGE_DIMENSIONS  = [10080, 14400, 18720, 24480]


# original manually written algos

GEMM1D_ALGORITHMS = [
    BenchmarkAlgorithm("AG_A_COL", gemm.AG_A_COL),
    BenchmarkAlgorithm("AG_A_ROW", gemm.AG_A_ROW),
    BenchmarkAlgorithm("AG_B_COL", gemm.AG_B_COL),
    BenchmarkAlgorithm("AG_B_ROW", gemm.AG_B_ROW),
    BenchmarkAlgorithm("RS_C_COL", gemm.RS_C_COL),
    BenchmarkAlgorithm("RS_C_ROW", gemm.RS_C_ROW),
]

GEMM2D_ALGORITHMS = [
    BenchmarkAlgorithm("AG_A_COL_AG_A_ROW", gemm.AG_A_COL_AG_A_ROW),
    BenchmarkAlgorithm("AG_A_COL_AG_B_COL", gemm.AG_A_COL_AG_B_COL),
    BenchmarkAlgorithm("AG_A_COL_AG_B_ROW", gemm.AG_A_COL_AG_B_ROW),
    BenchmarkAlgorithm("AG_A_COL_RS_C_COL", gemm.AG_A_COL_RS_C_COL),
    BenchmarkAlgorithm("AG_A_COL_RS_C_ROW", gemm.AG_A_COL_RS_C_ROW),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_COL", gemm.AG_A_ROW_AG_B_COL),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_ROW", gemm.AG_A_ROW_AG_B_ROW),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_COL", gemm.AG_A_ROW_RS_C_COL),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_ROW", gemm.AG_A_ROW_RS_C_ROW),
    BenchmarkAlgorithm("AG_B_COL_AG_B_ROW", gemm.AG_B_COL_AG_B_ROW),
    BenchmarkAlgorithm("AG_B_COL_RS_C_COL", gemm.AG_B_COL_RS_C_COL),
    BenchmarkAlgorithm("AG_B_COL_RS_C_ROW", gemm.AG_B_COL_RS_C_ROW),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_COL", gemm.AG_B_ROW_RS_C_COL),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_ROW", gemm.AG_B_ROW_RS_C_ROW),
    BenchmarkAlgorithm("RS_C_COL_RS_C_ROW", gemm.RS_C_COL_RS_C_ROW),
]


# 1d combiner

GEMM1D_PREV_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_PREV_COMPOSED", Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_PREV_COMPOSED", Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_PREV_COMPOSED", Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_PREV_COMPOSED", Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_PREV_COMPOSED", Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_PREV_COMPOSED", Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV).setup_and_run),
]

GEMM1D_NEXT_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_NEXT_COMPOSED", Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT).setup_and_run),
]

GEMM1D_ALL_COMPOSED = GEMM1D_PREV_COMPOSED + GEMM1D_NEXT_COMPOSED


GEMM2D_STANDARD_PREV_PREV_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_AG_A_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_B_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_RS_C_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
]

GEMM2D_STANDARD_PREV_NEXT_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_AG_A_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_B_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_RS_C_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
]

GEMM2D_STANDARD_NEXT_PREV_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_AG_A_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_B_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_RS_C_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
]

GEMM2D_STANDARD_NEXT_NEXT_COMPOSED = [
    BenchmarkAlgorithm("AG_A_COL_AG_A_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_AG_B_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_COL_RS_C_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_AG_B_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_A_ROW_RS_C_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_B_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_RS_C_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_RS_C_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_RS_C_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
]

GEMM2D_STANDARD_ALL_COMPOSED = (
    GEMM2D_STANDARD_PREV_PREV_COMPOSED +
    GEMM2D_STANDARD_NEXT_PREV_COMPOSED +
    GEMM2D_STANDARD_PREV_NEXT_COMPOSED +
    GEMM2D_STANDARD_NEXT_NEXT_COMPOSED
)

GEMM2D_REVERSED_PREV_PREV_COMPOSED = [
    BenchmarkAlgorithm("AG_A_ROW_AG_A_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_B_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_ROW_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_RS_C_COL_PREV_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
]

GEMM2D_REVERSED_PREV_NEXT_COMPOSED = [
    BenchmarkAlgorithm("AG_A_ROW_AG_A_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_B_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_ROW_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_RS_C_COL_NEXT_PREV_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
]

GEMM2D_REVERSED_NEXT_PREV_COMPOSED = [
    BenchmarkAlgorithm("AG_A_ROW_AG_A_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_B_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_ROW_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_PREV)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_RS_C_COL_PREV_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_PREV)).setup_and_run),
]

GEMM2D_REVERSED_NEXT_NEXT_COMPOSED = [
    BenchmarkAlgorithm("AG_A_ROW_AG_A_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_COL_AG_A_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_A_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_A_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_A_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.A, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("AG_B_ROW_AG_B_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.B, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_COL_AG_B_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_AG_B_ROW_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.B, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT)).setup_and_run),
    BenchmarkAlgorithm("RS_C_ROW_RS_C_COL_NEXT_NEXT_COMPOSED", Gemm2D(Gemm1D(MatrixCommunicated.C, SubtileScheme.ROW, CommunicationDirection.SEND_NEXT), Gemm1D(MatrixCommunicated.C, SubtileScheme.COL, CommunicationDirection.SEND_NEXT)).setup_and_run),
]

GEMM2D_REVERSED_ALL_COMPOSED = (
    GEMM2D_REVERSED_PREV_PREV_COMPOSED +
    GEMM2D_REVERSED_NEXT_PREV_COMPOSED +
    GEMM2D_REVERSED_PREV_NEXT_COMPOSED +
    GEMM2D_REVERSED_NEXT_NEXT_COMPOSED
)


# skips

GEMM1D_ALGORITHMS_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM1D_ALGORITHMS
]

GEMM2D_ALGORITHMS_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_ALGORITHMS
]

GEMM1D_PREV_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM1D_PREV_COMPOSED
]

GEMM1D_NEXT_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM1D_NEXT_COMPOSED
]

GEMM1D_ALL_COMPOSED_SKIP = GEMM1D_PREV_COMPOSED_SKIP + GEMM1D_NEXT_COMPOSED_SKIP

GEMM2D_STANDARD_PREV_PREV_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_STANDARD_PREV_PREV_COMPOSED
]

GEMM2D_STANDARD_NEXT_PREV_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_STANDARD_NEXT_PREV_COMPOSED
]

GEMM2D_STANDARD_PREV_NEXT_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_STANDARD_PREV_NEXT_COMPOSED
]

GEMM2D_STANDARD_NEXT_NEXT_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_STANDARD_NEXT_NEXT_COMPOSED
]

GEMM2D_STANDARD_ALL_COMPOSED_SKIP = (
    GEMM2D_STANDARD_PREV_PREV_COMPOSED_SKIP +
    GEMM2D_STANDARD_NEXT_PREV_COMPOSED_SKIP +
    GEMM2D_STANDARD_PREV_NEXT_COMPOSED_SKIP +
    GEMM2D_STANDARD_NEXT_NEXT_COMPOSED_SKIP
)

GEMM2D_REVERSED_PREV_PREV_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_REVERSED_PREV_PREV_COMPOSED
]

GEMM2D_REVERSED_NEXT_PREV_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_REVERSED_NEXT_PREV_COMPOSED
]

GEMM2D_REVERSED_PREV_NEXT_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_REVERSED_PREV_NEXT_COMPOSED
]

GEMM2D_REVERSED_NEXT_NEXT_COMPOSED_SKIP = [
    BenchmarkAlgorithm(algo.name + "_SKIP", functools.partial(algo.run_function, skip_computation=True))
    for algo in GEMM2D_REVERSED_NEXT_NEXT_COMPOSED
]

GEMM2D_REVERSED_ALL_COMPOSED_SKIP = (
    GEMM2D_REVERSED_PREV_PREV_COMPOSED_SKIP +
    GEMM2D_REVERSED_NEXT_PREV_COMPOSED_SKIP +
    GEMM2D_REVERSED_PREV_NEXT_COMPOSED_SKIP +
    GEMM2D_REVERSED_NEXT_NEXT_COMPOSED_SKIP
)

ALL_ALGORITHMS = (
    GEMM1D_ALGORITHMS +
    GEMM2D_ALGORITHMS +
    GEMM1D_ALL_COMPOSED +
    GEMM2D_STANDARD_ALL_COMPOSED +
    GEMM2D_REVERSED_ALL_COMPOSED
)

ALL_ALGORITHMS_SKIP = (
    GEMM1D_ALGORITHMS_SKIP +
    GEMM2D_ALGORITHMS_SKIP +
    GEMM1D_ALL_COMPOSED_SKIP +
    GEMM2D_STANDARD_ALL_COMPOSED_SKIP +
    GEMM2D_REVERSED_ALL_COMPOSED_SKIP
)


def warmup(comm):
    warmup_matrix = np.random.rand(4096, 4096).astype(MATRIX_DTYPE)
    for _ in range(10):
        np.matmul(warmup_matrix, warmup_matrix)
    del warmup_matrix
    gc.collect()
    comm.Barrier()


if __name__ == "__main__":
    comm, size, rank = mpi_setup()
    warmup(comm)

    nodes = int(os.environ.get("SLURM_NNODES", 1))
    ntasks_per_node = size // nodes

    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="+", required=True)
    parser.add_argument("--dimensions", required=True, choices=["SMALL", "MEDIUM", "LARGE"])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    current_module = sys.modules[__name__]
    algorithms = []
    for group_name in args.algos:
        algorithms.extend(getattr(current_module, group_name))

    dimensions = getattr(current_module, args.dimensions + "_DIMENSIONS")
    grid_configurations = [(px, size // px) for px in range(1, size + 1) if size % px == 0]

    num_algorithms = len(algorithms)
    num_dimension_combos = len(dimensions) ** 3
    num_grid_configurations = len(grid_configurations)
    total_runs = num_algorithms * num_dimension_combos * num_grid_configurations * args.runs

    if rank == 0:
        print(f"Algorithms:            {num_algorithms}")
        print(f"Dimension combos:      {num_dimension_combos}")
        print(f"Grid configurations:   {num_grid_configurations}")
        print(f"Runs per combo:        {args.runs}")
        print(f"Total benchmark runs:  {total_runs}")

    if rank == 0:
        os.makedirs(BENCHMARK_FOLDER, exist_ok=True)
        existing_ids = [
            int(f.stem.removeprefix("benchmark_"))
            for f in BENCHMARK_FOLDER.glob("benchmark_*.csv")
            if f.stem.removeprefix("benchmark_").isdigit()
        ]
        next_id = max(existing_ids, default=0) + 1
        output_path = BENCHMARK_FOLDER / f"benchmark_{next_id:03d}.csv"
        csv_file = open(output_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=["algorithm", "m", "k", "n", "px", "py", "run_index", "elapsed_time", "correct", "timestamp", "world_size", "nodes", "ntasks_per_node"])
        writer.writeheader()

    try:
        progress_output = open('/dev/tty', 'w') if rank == 0 else None
    except OSError:
        progress_output = sys.stderr if rank == 0 else None
    with tqdm(total=total_runs, disable=(rank != 0), file=progress_output) as progress_bar:
        for px, py in grid_configurations:
            for m, k, n in itertools.product(dimensions, repeat=3):
                for algo in algorithms:
                    for run_index in range(args.runs):
                        progress_bar.set_description(algo.name)
                        result = algo.run_function(m, k, n, px, py)
                        if rank == 0:
                            writer.writerow({
                                "algorithm": algo.name,
                                "m": m, "k": k, "n": n,
                                "px": px, "py": py,
                                "run_index": run_index,
                                "elapsed_time": result["elapsed_time"],
                                "correct": result["correct"],
                                "timestamp": datetime.now().isoformat(),
                                "world_size": size,
                                "nodes": nodes,
                                "ntasks_per_node": ntasks_per_node,
                            })
                        del result
                        progress_bar.update(1)
                    gc.collect()

    if progress_output is not None:
        progress_output.close()

    if rank == 0:
        csv_file.close()
        print(f"Results written to {output_path}")
