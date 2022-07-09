#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf_ms
#SBATCH --output=pf_ms.out
#SBATCH --error=pf_ms.err
#SBATCH --time=6-23:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --array=1-100

# Load Python modules
module load python/booth/3.8/3.8.5

scratch_dir="/scratch/${SLURM_JOB_USER}/pf_ms/"
mkdir -p $scratch_dir

srun python3 Run_Aso1_0_CPU_$SLURM_ARRAY_TASK_ID.py  