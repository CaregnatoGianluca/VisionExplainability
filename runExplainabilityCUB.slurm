#!/bin/bash -l
#SBATCH --job-name=ResNet_ExplainabilityCUB
#SBATCH --output=ResNet_ExplainabilityCUB_%j.out
#SBATCH --error=ResNet_ExplainabilityCUB_%j.err
#SBATCH --mail-user=gianluca.caregnato@studenti.unipd.it
#SBATCH --mail-type=END,FAIL
#SBATCH --time=30:00:00
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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
  python explainabilityCUB.py
'
