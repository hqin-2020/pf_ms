#!/bin/bash

#SBATCH --job-name=pf_ms
#SBATCH --output=pf_ms.out
#SBATCH --error=pf_ms.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=28
#SBATCH --mem=60G
#SBATCH --array=1-100

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_$SLURM_ARRAY_TASK_ID.py  