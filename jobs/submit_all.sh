#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")

TASK_COUNTS=(2 4 6 8 10 12 16 20 24 36 48 60 72 96 120)

ALGO_LABELS=("original" "1d_composed" "2d_standard" "2d_reversed" "original_skip" "1d_composed_skip" "2d_standard_skip" "2d_reversed_skip")
ALGO_VALUES=(
    "GEMM1D_ALGORITHMS GEMM2D_ALGORITHMS"
    "GEMM1D_ALL_COMPOSED"
    "GEMM2D_STANDARD_ALL_COMPOSED"
    "GEMM2D_REVERSED_ALL_COMPOSED"
    "GEMM1D_ALGORITHMS_SKIP GEMM2D_ALGORITHMS_SKIP"
    "GEMM1D_ALL_COMPOSED_SKIP"
    "GEMM2D_STANDARD_ALL_COMPOSED_SKIP"
    "GEMM2D_REVERSED_ALL_COMPOSED_SKIP"
)

mkdir -p "$SCRIPT_DIR/logs"

for total_tasks in "${TASK_COUNTS[@]}"; do
    nodes=$(( (total_tasks + 23) / 24 ))
    ntasks_per_node=$(( total_tasks / nodes ))

    for i in "${!ALGO_LABELS[@]}"; do
        label="${ALGO_LABELS[$i]}"
        groups="${ALGO_VALUES[$i]}"
        job_name="bench_${total_tasks}t_${label}"

        ALGO_GROUPS="$groups" sbatch \
            --nodes=$nodes \
            --ntasks-per-node=$ntasks_per_node \
            --job-name="$job_name" \
            --output="$SCRIPT_DIR/logs/${job_name}_%j.out" \
            --error="$SCRIPT_DIR/logs/${job_name}_%j.err" \
            --export=ALL \
            "$SCRIPT_DIR/benchmark.sbatch"
    done
done
