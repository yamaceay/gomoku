#!/bin/bash
#SBATCH --job-name=${train_pre}
#SBATCH --partition=${gpu_2h}
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv gomoku.sif python -m fit \
    --START=0 --END=1000 --STEP=100 \ 
    --DIR=_dens --PLAYER=dense \ 
    --LR=0.01 --LR_DECAY=0.995