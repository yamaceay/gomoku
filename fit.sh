#!/bin/bash
#SBATCH --job-name=${train_pre}
#SBATCH --partition=${gpu_2h}
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j

apptainer run --nv gomoku.sif python -m src.mcts_adp
# apptainer run --nv gomoku.sif python -m test
# apptainer run --nv gomoku.sif python -m fit --DIR="_dens" --PLAYER="dense" --START=0 --END=1000 --STEP=10 --BATCH_SIZE=80 --LR=0.01 --LR_DECAY=0.995 --NO_EVAL
