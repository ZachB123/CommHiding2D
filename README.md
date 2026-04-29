# 2D Communication-Hiding GEMM

## Overview

This project implements communication-hiding General Matrix Multiply (GEMM) algorithms for distributed computing using MPI. Each algorithm computes **C = A x B + C** distributed across a 1D or 2D processor grid, overlapping non-blocking communication with local computation to hide latency.

There are three implementation layers:

- `gemm.py` - original, per-algorithm explicit implementations (not actively used)
- `refactored_gemm.py` - shared infrastructure with cleaner code; 15 two-dimensional and 6 one-dimensional algorithms
- `composed_gemm_1d.py` / `composed_gemm_2d.py` - configuration-driven classes (`Gemm1D`, `Gemm2D`) that automatically construct algorithm variants from parameters

## Requirements

- Python 3.6 or higher
- An MPI implementation (e.g., OpenMPI or MPICH)

## Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running a Single Algorithm

`driver.py` supports the 15 manually written 2D algorithms and 6 1D algorithms:

```bash
mpirun -n <num_processes> python driver.py -a <algorithm> -m <M> -k <K> -n <N> -px <PX> -py <PY>
```

| Argument      | Short Flag | Description                                     | Required |
|---------------|------------|-------------------------------------------------|----------|
| `--algorithm` | `-a`       | Name of the GEMM algorithm to run              | Yes      |
| `--m`         | `-m`       | Number of rows in matrix A (and C)             | Yes      |
| `--k`         | `-k`       | Number of columns in A / rows in B             | Yes      |
| `--n`         | `-n`       | Number of columns in matrix B (and C)          | Yes      |
| `--px`        | `-px`      | Number of processors in the first dimension    | Yes      |
| `--py`        | `-py`      | Number of processors in the second dimension   | Yes      |

Use `--oversubscribe` with `mpirun` when testing locally with more processes than physical cores.

**Example:**
```bash
mpirun -n 8 python driver.py -a AG_A_ROW_AG_B_ROW -m 256 -k 256 -n 256 -px 4 -py 2
```

### Running Tests

`tests.py` covers all algorithm variants including composed ones.

**Test all algorithms:**
```bash
mpirun -n 8 python tests.py
```

**Test a specific algorithm:**
```bash
mpirun -n 8 python tests.py -a AG_A_COL_AG_A_ROW_PREV_PREV_COMPOSED
```

**Test across multiple process counts (1 to 12):**
```bash
./test_runner.sh

# Single algorithm across all process counts
./test_runner.sh -a AG_A_ROW_AG_B_ROW
```

## Algorithm Naming Convention

Algorithm names encode the communication operations performed:

- `AG` = AllGather, `RS` = ReduceScatter
- `A` / `B` / `C` = which matrix is communicated
- `ROW` / `COL` = subtile scheme (how the matrix is split)

2D algorithms compose two 1D operations separated by `_`. The first operation is the outer loop and the second is the inner loop:

```
AG_A_ROW_AG_B_ROW   -- outer: allgather A by row, inner: allgather B by row
RS_C_COL_RS_C_ROW   -- outer: reduce-scatter C by column, inner: reduce-scatter C by row
```

**Composed variants** (produced by `Gemm1D` / `Gemm2D`) append a direction suffix and `_COMPOSED`:

- 1D composed: `<ALGORITHM>_PREV_COMPOSED` or `<ALGORITHM>_NEXT_COMPOSED`
- 2D composed: `<ALGORITHM>_<DIR1>_<DIR2>_COMPOSED` where DIR1 and DIR2 are `PREV` or `NEXT`
- Reversed 2D: the outer and inner loop operations are swapped relative to the canonical ordering

`PREV` means the rank sends data toward lower-ranked neighbors; `NEXT` means toward higher-ranked neighbors.

## Available Algorithms

### Manually Written 2D Algorithms (px x py grid)

These are implemented explicitly in `refactored_gemm.py`:

| # | Algorithm |
|---|-----------|
| 1 | `AG_A_COL_AG_A_ROW` |
| 2 | `AG_A_COL_AG_B_COL` |
| 3 | `AG_A_COL_AG_B_ROW` |
| 4 | `AG_A_COL_RS_C_COL` |
| 5 | `AG_A_COL_RS_C_ROW` |
| 6 | `AG_A_ROW_AG_B_COL` |
| 7 | `AG_A_ROW_AG_B_ROW` |
| 8 | `AG_A_ROW_RS_C_COL` |
| 9 | `AG_A_ROW_RS_C_ROW` |
| 10 | `AG_B_COL_AG_B_ROW` |
| 11 | `AG_B_COL_RS_C_COL` |
| 12 | `AG_B_COL_RS_C_ROW` |
| 13 | `AG_B_ROW_RS_C_COL` |
| 14 | `AG_B_ROW_RS_C_ROW` |
| 15 | `RS_C_COL_RS_C_ROW` |

### 1D Algorithms (linear layout)

`AG_A_COL`, `AG_A_ROW`, `AG_B_COL`, `AG_B_ROW`, `RS_C_COL`, `RS_C_ROW`

### Composed 1D Algorithms (`Gemm1D`)

Each of the 6 1D operations with `PREV` and `NEXT` directions (12 total):

`AG_A_COL_PREV_COMPOSED`, `AG_A_COL_NEXT_COMPOSED`, `AG_A_ROW_PREV_COMPOSED`, `AG_A_ROW_NEXT_COMPOSED`, `AG_B_COL_PREV_COMPOSED`, `AG_B_COL_NEXT_COMPOSED`, `AG_B_ROW_PREV_COMPOSED`, `AG_B_ROW_NEXT_COMPOSED`, `RS_C_COL_PREV_COMPOSED`, `RS_C_COL_NEXT_COMPOSED`, `RS_C_ROW_PREV_COMPOSED`, `RS_C_ROW_NEXT_COMPOSED`

### Composed 2D Algorithms (`Gemm2D`)

All 15 canonical 2D pairings (same set as the manually written algorithms) are available with all 4 direction combinations (`PREV_PREV`, `NEXT_PREV`, `PREV_NEXT`, `NEXT_NEXT`), giving 60 variants. The reversed set (outer and inner loops swapped) adds another 60 variants (120 composed 2D total).

Example names:
```
AG_A_COL_AG_A_ROW_PREV_PREV_COMPOSED
AG_B_ROW_RS_C_COL_NEXT_PREV_COMPOSED
RS_C_ROW_RS_C_COL_NEXT_NEXT_COMPOSED   # reversed of RS_C_COL_RS_C_ROW
```

## Project Structure

```
.
├── driver.py              # Entry point for running individual algorithms
├── tests.py               # Test suite for all algorithm variants
├── test_runner.sh         # Runs tests across process counts 1 to 12
├── composed_gemm_1d.py    # Gemm1D: config-driven 1D algorithm class
├── composed_gemm_2d.py    # Gemm2D: config-driven 2D algorithm class
├── refactored_gemm.py     # Manually written 2D and 1D implementations
├── gemm.py                # Original per-algorithm implementations
├── data_classes.py        # Frozen dataclasses (DivisibiltyRequirements, DistributionFunctions, etc.)
├── enums.py               # Enums (GemmDimension, MatrixCommunicated, SubtileScheme, etc.)
├── distribution.py        # Matrix distribution strategies
├── util.py                # Buffers, MPI helpers, matrix generation
├── communicator.py        # MPI sub-communicator construction
├── debug.py               # MPI-aware logging utilities
├── constants.py           # Global constants and implementation toggle
├── true_order.txt         # Canonical ordering of the 15 base 2D algorithm names
└── requirements.txt       # Python dependencies
```

## Algorithm Details

Each algorithm computes: **C = A x B + C**

- A is M x K
- B is K x N
- C is M x N

All algorithms overlap communication with computation using non-blocking MPI operations (`Isend` / `Irecv`). Data is sent before the local matmul begins, so the network transfer runs in parallel with computation and `Waitall` only blocks if the transfer has not yet finished.

### Execution Flow

1. Generate deterministic matrices (seed=42) and distribute local tiles per the algorithm's distribution scheme
2. Create two MPI sub-communicators: `nearby_rank_communicator` (contiguous ranks) and `remainder_communicator` (modulo groupings)
3. Outer loop (first communicator) x inner loop (second communicator): initiate `Isend`/`Irecv`, compute local matmul, then `Waitall`
4. Gather C tiles and compare against `np.matmul(A, B) + C`

### Dimension Requirements

Each algorithm has minimum dimension requirements based on `px`, `py`, and total process count. The `Gemm2DConfig.assert_divisibility` method enforces these before the algorithm runs.

## Configuration

### Switching Implementations

Edit `constants.py` to select between `refactored_gemm.py` and `gemm.py` for `driver.py`:

```python
USE_REFACTORED_ALGORITHMS = True   # use refactored_gemm.py
USE_REFACTORED_ALGORITHMS = False  # use gemm.py
```

This flag does not affect `Gemm1D` or `Gemm2D` in `tests.py`.

### Matrix Data Type

```python
MATRIX_DTYPE = np.float32  # single precision
MATRIX_DTYPE = np.float64  # double precision
```

## Testing

The test suite runs 30 random-parameter iterations per algorithm. For each iteration it:

- Picks a random `px` / `py` factoring of the process count
- Scales minimum required dimensions by a random multiplier (1 to 50)
- Runs the algorithm and compares output to `np.matmul(A, B) + C`

All 153 algorithm variants (15 manual 2D + 6 1D + 12 composed 1D + 60 main composed 2D + 60 reversed composed 2D) can be run individually with `-a` or all at once.
