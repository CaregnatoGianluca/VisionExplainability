#!/bin/bash -l
#SBATCH --job-name=CNN_TrainingKDEF
#SBATCH --output=CNN_TrainingKDEF_%j.out
#SBATCH --error=CNN_TrainingKDEF_%j.err
#SBATCH --mail-user=gianluca.caregnato@studenti.unipd.it
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=10:00:00
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -x

# 1. Scratch Setup
SCRATCH=/ext/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
export TMP="$SCRATCH"
export TEMP="$SCRATCH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

singularity exec -B /nfsd --nv /nfsd/opt/sif-images/tensorflow_latest-gpu-new.sif bash -lc '
  source /nfsd/nannidl/caregnatog/venvs/mamba/bin/activate &&
  cd /nfsd/nannidl/caregnatog/VisionExplainability/CNN &&
  python training.py
'
