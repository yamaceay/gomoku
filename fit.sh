#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=cpu-7d
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j
# #SBATCH --gpus-per-node=1

# training
apptainer run --nv gomoku.sif python -m src.train -g M

# evaluation
# apptainer run --nv gomoku.sif python -m src.comp -g M
