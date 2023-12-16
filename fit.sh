#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j

# # testing
# apptainer run --nv gomoku.sif python -m test --DIR="_dens" --PLAYER="dense" --EPOCH=2500 --EPSILON=.1 --ITERATIONS=10000

# # training
apptainer run --nv gomoku.sif python -m fit --DIR="_dens" --PLAYER="dense" --START=320 --END=500 --STEP=10 --BATCH_SIZE=20 --LR=0.01 --LR_DECAY=0.995 --NO_EVAL --EPSILON=0.25