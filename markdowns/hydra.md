#### Login
`ssh hydra`

#### Define Build Step
```bash
Bootstrap: docker
From: python:3.11

%files
    $PWD/requirements.txt requirements.txt

%post
    pip install --root-user-action=ignore -r requirements.txt
```

#### Build Step
`apptainer build gomoku.sif gomoku.def`

#### Run Step
`apptainer run --nv gomoku.sif ${CMD}`

#### Connect to a Partition
`srun --partition=${PARTITION} --pty bash`

#### Define a Job
```bash
#!/bin/bash
#SBATCH --job-name=${name_training}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv gomoku.sif ${COMMAND}
```

#### Run a Job
`sbatch --mail-user=${MAILTO}@tu-berlin.de ${command}`

#### Monitor the Status of Jobs
`watch -c "squeue -u ${USERNAME}"`

#### Monitor a File
`tail -f ${LOG_FILE}`

#### End a Job
`scancel ${JOB_ID}`