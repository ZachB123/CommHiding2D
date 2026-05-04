#!/bin/bash
#SBATCH -A gts-rvuduc3
#SBATCH -q inferno
#SBATCH --job-name=2d_comm_hiding_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

module load anaconda3

JOBS_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$JOBS_DIR/..

mkdir -p $JOBS_DIR/logs
pip install -r $PROJECT_ROOT/requirements.txt

cd $PROJECT_ROOT/src

srun python benchmark.py \
    --algos GEMM1D_ALGORITHMS \
    --dimensions SMALL \
    --runs 1
