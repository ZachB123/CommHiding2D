# CLAUDE.md

The background reading is in CommHidingThesis.pdf, this is very important to understand

Be very cautious about what programs you run and make sure you force exit them after a short period of time as it is very easy for tests to get completely out of hand and take several hours to finish running.

## Commands

**Run tests (all algorithms, specific process count):**
```bash
mpirun -n 8 python tests.py
```

**Run tests (specific algorithm):**
```bash
mpirun -n 8 python tests.py -a AG_A_ROW_AG_B_ROW
```

**Run test suite across multiple process counts:**
```bash
./test_runner.sh
./test_runner.sh -a AG_A_ROW_AG_B_ROW
```

**Run a single algorithm:**
```bash
mpirun -n 8 python driver.py -a AG_A_ROW_AG_B_ROW -m 256 -k 256 -n 256 -px 4 -py 2
```

## Architecture

This project implements 2D communication-hiding GEMM (C = A×B + C) distributed across MPI processor grids, where non-blocking communication overlaps with computation to hide latency.

### Three implementation layers (selected via `constants.py`)

- `gemm.py` — original, per-algorithm explicit implementations; not needed, do not use.
- `refactored_gemm.py` — shared infrastructure, cleaner code
- `composed_gemm.py` — current/active: configuration-driven, parameterized 1D and 2D classes


### Algorithm naming convention

Algorithm names encode what is communicated and how:
- `AG` = AllGather, `RS` = ReduceScatter
- `A/B/C` = which matrix
- `ROW/COL` = communication direction
- Two operations separated by `_` indicate outer × inner loop communication (e.g., `AG_A_ROW_AG_B_ROW`)

**2D algorithms** require a `px × py` processor grid. **1D algorithms** (e.g., `AG_A_ROW`) use a linear layout.

### Execution flow per algorithm

1. Generate deterministic matrices (seed=42), distribute local tiles per algorithm's distribution scheme
2. Create two MPI sub-communicators (nearby-rank and remainder groupings)
3. Outer loop (first communicator) × inner loop (second communicator):
   - Initiate non-blocking `Isend`/`Irecv` → compute local matmul → `Waitall`
4. Gather C tiles, compare against `np.matmul(A, B) + C` reference

### Key support files

- `distribution.py` — distribution strategies (row_major, col_major, pure_row, pure_col, block_cyclic, etc.)
- `util.py` — `DoubleBuffer` (pipeline rotation), `SubtileBuffer`, `AccumulationBuffer`, MPI timing
- `communicator.py` — `nearby_rank_communicator` (contiguous ranks) and `remainder_communicator` (modulo groupings)
- `enums.py` / `data_classes.py` — `GemmDimension`, `MatrixCommunicated`, `SubtileScheme`, `CommunicationDirection`, frozen config dataclasses
- `debug.py` — MPI-aware logging (`rank_print`, `parallel_print`)
- `constants.py` — `USE_REFACTORED_ALGORITHMS`, `MATRIX_DTYPE`, `MPI_DTYPE`

### Testing

`tests.py` runs 30 random-parameter iterations per algorithm across all factor-pairs of the MPI process count. Dimensions are multiplied by random(1–50) from per-algorithm minimums.

# Goal
The goal is to build a modular, configuration-driven library that automatically constructs combinations of 2D algorithms, rather than manually implementing each variant as in `refactored_gemm.py`.

### Naming

2D algorithms are composed of two 1D algorithms. For each component, we specify which matrix to communicate, its distribution, and the direction data is sent in the ring.

The full name is constructed as follows (AG applies to A and B matrices; RS applies to C):

`<Matrix>_<Distribution>_<Direction>_<Matrix>_<Distribution>_<Direction>`

- Matrix options: `AG_A`, `AG_B`, `RS_C`
- Distribution options: `ROW`, `COL`
- Direction options: `NEXT`, `PREV` — indicates which direction data travels in the ring (send to next rank vs. previous rank). Direction is only used for composed algorithms.

The naming convention is not yet fully locked in and has some inconsistency — this should be resolved. There should be a clear way to distinguish manually written algorithms from dynamically composed ones.

