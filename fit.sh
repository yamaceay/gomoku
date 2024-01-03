#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=cpu-2d
# #SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j

# # testing
# apptainer run --nv gomoku.sif python -m test --DIR="_dens" --PLAYER="dense" --EPOCH=10000 --EPSILON=.1 --ITERATIONS=200

# # testing adp
# apptainer run --nv gomoku.sif python -m test_adp

# # training
apptainer run --nv gomoku.sif python -m src.train

# # training
# apptainer run --nv gomoku.sif python -m fit --DIR="_conv" --PLAYER="conv" --START=0 --END=10000 --STEP=25 --BATCH_SIZE=5 --CHECKPOINT=100 --LR=0.01 --LR_DECAY=0.998 --NO_EVAL --EPSILON=0.25
