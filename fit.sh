#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j
# #SBATCH --gpus-per-node=1

if [ $1 == "train" ]; then
    apptainer run --nv gomoku.sif python -m src.train ${@:2}
elif [ $1 == "eval" ]; then
    apptainer run --nv gomoku.sif python -m src.comp ${@:2}
elif [ $1 == "stats" ]; then
    apptainer run --nv gomoku.sif python -m src.stats ${@:2}	
else
    echo "Usage: sbatch fit.sh <train|eval|stats> …"
    exit 1
fi
