#!/bin/bash

MAX_PROCESSES=36

VALID_ALGORITHMS=(
    "AG_A_COL_AG_A_ROW"
    "AG_A_COL_AG_B_COL"
    "AG_A_COL_AG_B_ROW"
    "AG_A_COL_RS_C_COL"
    "AG_A_COL_RS_C_ROW"
    "AG_A_ROW_AG_B_COL"
    "AG_A_ROW_AG_B_ROW"
    "AG_A_ROW_RS_C_COL"
    "AG_A_ROW_RS_C_ROW"
    "AG_B_COL_AG_B_ROW"
    "AG_B_COL_RS_C_COL"
    "AG_B_COL_RS_C_ROW"
    "AG_B_ROW_RS_C_COL"
    "AG_B_ROW_RS_C_ROW"
    "RS_C_COL_RS_C_ROW"
    "AG_A_COL"
    "AG_A_ROW"
    "AG_B_COL"
    "AG_B_ROW"
    "RS_C_COL"
    "RS_C_ROW"
)

is_valid_algorithm() {
    local algo="$1"
    for valid_algo in "${VALID_ALGORITHMS[@]}"; do
        if [ "$algo" == "$valid_algo" ]; then
            return 0
        fi
    done
    return 1
}

ALGORITHM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--algorithm)
            ALGORITHM="$2"
            if ! is_valid_algorithm "$ALGORITHM"; then
                echo "Error: Invalid algorithm '$ALGORITHM'"
                echo "Valid algorithms are:"
                printf '  %s\n' "${VALID_ALGORITHMS[@]}"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-a|--algorithm <algorithm_name>]"
            exit 1
            ;;
    esac
done

# Build python command
if [ -n "$ALGORITHM" ]; then
    PYTHON_CMD="python tests.py -a $ALGORITHM"
    echo "Running tests for algorithm: $ALGORITHM"
else
    PYTHON_CMD="python tests.py"
    echo "Running all algorithm tests"
fi

echo "========================================"

for n in $(seq 1 $MAX_PROCESSES); do
    echo "Running with n = $n"
    mpirun --oversubscribe -n "$n" $PYTHON_CMD
    echo "----------------------------------------"
done