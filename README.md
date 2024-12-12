# 2D Comm Hidden Gemm

## Overview
This project provides a Python implementation to run various Communication Hiding General Matrix Multiply (GEMM) algorithms over a 2d grid of processors.

## Requirements
- Python 3.6 or higher
- An MPI implementation (e.g., OpenMPI)


## Installation
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure that an MPI implementation is installed on your system (e.g., `openmpi` or `mpich`).

## Usage
Run the Python program with the following options:

### Command-line Arguments
| Argument         | Short Flag | Description                                       | Required | Example          |
|------------------|------------|---------------------------------------------------|----------|------------------|
| `--algorithm`    | `-a`       | The name of the GEMM algorithm to run            | Yes      | `-a algorithm1`  |
| `--m`            | `-m`       | The value for M dimension                        | Yes      | `-m 128`         |
| `--k`            | `-k`       | The value for K dimension                        | Yes      | `-k 128`         |
| `--n`            | `-n`       | The value for N dimension                        | Yes      | `-n 128`         |
| `--px`           | `-px`      | Number of processors in the first dimension      | Yes      | `-px 4`          |
| `--py`           | `-py`      | Number of processors in the second dimension     | Yes      | `-py 2`          |

### Example
To run the program:
```bash
mpirun -n 8 python driver.py -a AG_A_ROW_X_AG_B_ROW -m 128 -k 128 -n 128 -px 4 -py 2
```

### Explanation of the Example
- `mpirun -np 8`: Runs the program using 8 processes.
- `python driver.py`: Specifies the Python file to execute.
- `-a AG_A_ROW_X_AG_B_ROW`: Uses the algorithm `AG_A_ROW_X_AG_B_ROW`.
- `-m 128 -k 128 -n 128`: Sets matrix dimensions M, K, and N to 128 each.
- `-px 4 -py 2`: Divides the processors into a 4x2 grid.

## Testing
To test all of the algorithms run
```bash
mpirun -n 8 python test.py
```

## Available Algorithms
The following algorithms are supported:
- `AG_A_COL_AG_A_ROW`
- `AG_A_COL_AG_B_COL`
- `AG_A_COL_AG_B_ROW`
- `AG_A_COL_RS_C_COL`
- `AG_A_COL_RS_C_ROW`
- `AG_A_ROW_AG_B_COL`
- `AG_A_ROW_AG_B_ROW`
- `AG_A_ROW_RS_C_COL`
- `AG_A_ROW_RS_C_ROW`
- `AG_B_COL_AG_B_ROW`
- `AG_B_COL_RS_C_COL`
- `AG_B_COL_RS_C_ROW`
- `AG_B_ROW_RS_C_COL`
- `AG_B_ROW_RS_C_ROW`
- `RS_C_COL_RS_C_ROW`





