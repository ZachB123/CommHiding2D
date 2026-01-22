# 2D Communication-Hiding GEMM

## Overview
This project implements different Communication-Hiding General Matrix Multiply (GEMM) algorithms for distributed computing using MPI. The algorithms distribute matrix multiplication across a 2D or 1D processor grid, overlapping computation with communication to improve performance.

## Requirements
- Python 3.6 or higher
- An MPI implementation (e.g., OpenMPI or MPICH)

## Installation
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure MPI is installed on your system:

## Usage

### Running a Single Algorithm
Execute a specific GEMM algorithm with custom parameters:

```bash
mpirun -n <num_processes> python driver.py -a <algorithm> -m <M> -k <K> -n <N> -px <PX> -py <PY>
```

**Command-line Arguments:**
| Argument      | Short Flag | Description                                    | Required | Example         |
|---------------|------------|------------------------------------------------|----------|-----------------|
| `--algorithm` | `-a`       | Name of the GEMM algorithm to run             | Yes      | `-a AG_A_ROW_AG_B_ROW` |
| `--m`         | `-m`       | Number of rows in matrix A (and C)            | Yes      | `-m 128`        |
| `--k`         | `-k`       | Number of columns in A and rows in B            | Yes      | `-k 128`        |
| `--n`         | `-n`       | Number of columns in matrix B (and C)         | Yes      | `-n 128`        |
| `--px`        | `-px`      | Number of processors in first dimension       | Yes      | `-px 4`         |
| `--py`        | `-py`      | Number of processors in second dimension      | Yes      | `-py 2`         |

**Example:**
```bash
mpirun -n 8 python driver.py -a AG_A_ROW_AG_B_ROW -m 256 -k 256 -n 256 -px 4 -py 2
```

This runs the `AG_A_ROW_AG_B_ROW` algorithm with 256×256 matrices on an 8-processor (4×2) grid. Optionally you can use --oversubscribe with the mpirun command if you are testing locally and do not have the number of processors on your machine that you requested.

### Running Tests

**Test All Algorithms:**
```bash
mpirun -n 8 python tests.py
```

**Test a Specific Algorithm:**
```bash
mpirun -n 8 python tests.py -a AG_A_ROW_AG_B_ROW
```

**Comprehensive Testing Script:**
The `test_runner.sh` script tests algorithms across different processor counts:
```bash
# Test all algorithms with 1 to 36 processes
./test_runner.sh

# Test specific algorithm with varying process counts
./test_runner.sh -a AG_A_ROW_AG_B_ROW
```

## Available Algorithms

### Two-Dimensional Algorithms (2D Grid: px × py)
These algorithms distribute work across a 2D processor grid where `px * py = total_processes`.

- `AG_A_COL_AG_A_ROW` - Allgather A by column, then by row
- `AG_A_COL_AG_B_COL` - Allgather A by column and B by column
- `AG_A_COL_AG_B_ROW` - Allgather A by column and B by row
- `AG_A_ROW_AG_B_COL` - Allgather A by row and B by column
- `AG_A_ROW_AG_B_ROW` - Allgather A by row and B by row
- `AG_B_COL_AG_B_ROW` - Allgather B by column, then by row
- `AG_A_COL_RS_C_COL` - Allgather A by column, reduce-scatter C by column
- `AG_A_COL_RS_C_ROW` - Allgather A by column, reduce-scatter C by row
- `AG_A_ROW_RS_C_COL` - Allgather A by row, reduce-scatter C by column
- `AG_A_ROW_RS_C_ROW` - Allgather A by row, reduce-scatter C by row
- `AG_B_COL_RS_C_COL` - Allgather B by column, reduce-scatter C by column
- `AG_B_COL_RS_C_ROW` - Allgather B by column, reduce-scatter C by row
- `AG_B_ROW_RS_C_COL` - Allgather B by row, reduce-scatter C by column
- `AG_B_ROW_RS_C_ROW` - Allgather B by row, reduce-scatter C by row
- `RS_C_COL_RS_C_ROW` - Reduce-scatter C by column, then by row

### One-Dimensional Algorithms (1D Linear Layout)
These algorithms use a linear processor arrangement (effectively px=1 or py=1).

- `AG_A_COL` - Allgather matrix A by column
- `AG_A_ROW` - Allgather matrix A by row
- `AG_B_COL` - Allgather matrix B by column
- `AG_B_ROW` - Allgather matrix B by row
- `RS_C_COL` - Reduce-scatter result C by column
- `RS_C_ROW` - Reduce-scatter result C by row

## Project Structure

```
.
├── driver.py              # Main entry point for running algorithms
├── tests.py               # Comprehensive test suite
├── test_runner.sh         # Script for testing across processor counts
├── gemm_refactored.py     # Refactored GEMM algorithm implementations
├── gemm.py                # Original GEMM implementations
├── util.py                # Utility functions (matrix generation, MPI helpers)
├── distribution.py        # Matrix distribution strategies
├── communicator.py        # MPI communicator creation helpers
├── debug.py               # Debugging and logging utilities
├── constants.py           # Global constants and configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Algorithm Details

Each algorithm computes: **C = A × B + C**

Where:
- A is an M × K matrix
- B is a K × N matrix  
- C is an M × N matrix

The algorithms differ in:
1. **Matrix Distribution**: How A, B, and C are initially distributed across processors
2. **Communication Pattern**: Which matrices are communicated and in what order
3. **Computational Strategy**: How local multiplications are organized

### Communication Hiding
All algorithms overlap communication with computation using non-blocking MPI operations (`Isend`/`Irecv`), allowing processors to begin sending/receiving data while performing local matrix multiplications.

## Configuration

### Switching Between Implementations
Edit `constants.py` to toggle between implementations:

```python
USE_REFACTORED_ALGORITHMS = True  # Use refactored version
USE_REFACTORED_ALGORITHMS = False # Use original version
```

### Matrix Data Type
Modify `constants.py` to change floating-point precision:

```python
MATRIX_DTYPE = np.float32  # Single precision
MATRIX_DTYPE = np.float64  # Double precision
```

## Testing

The test suite (`tests.py`) automatically:
- Tests algorithms with random matrix dimensions
- Validates correctness against NumPy's `matmul`
- Runs 30 iterations per algorithm with varying parameters
- Ensures compatibility with different processor grid configurations

**Test Configuration:**
- Each algorithm has specific minimum dimension requirements based on px, py, and size
- Dimensions are randomized within valid ranges
- Results are verified using floating-point comparison with tolerance





