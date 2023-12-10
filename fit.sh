#!/bin/bash
#SBATCH --job-name=${train_pre}
#SBATCH --partition=${gpu_2h}
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv gomoku.sif python -m fit \
    --DIR="_dens" --PLAYER="pre" \ 
    --START=0 --END=50 --STEP=5 \
    --BATCH_SIZE=250 \ 
    --LR=0.01 --LR_DECAY=0.995