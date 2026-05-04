#!/bin/bash
#SBATCH -A gts-rvuduc3
#SBATCH -q inferno
#SBATCH --time=24:00:00

module load anaconda3

JOBS_DIR=$PROJECT_ROOT/jobs

mkdir -p $JOBS_DIR/logs
pip install -r $PROJECT_ROOT/requirements.txt

cd $PROJECT_ROOT/src

srun python benchmark.py \
    --algos $ALGO_GROUPS \
    --dimensions ${DIMENSIONS:-LARGE} \
    --runs ${RUNS:-3}
