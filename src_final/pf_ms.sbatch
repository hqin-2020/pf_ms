#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf_ms_final
#SBATCH --output=pf_ms_final.out
#SBATCH --error=pf_ms_final.err
#SBATCH --time=1-23:00:00
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 Final_distbution.py  