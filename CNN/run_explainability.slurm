#!/bin/bash -l

#SBATCH --job-name=resnet_train
#SBATCH --output=resnet_train_%j.out
#SBATCH --error=resnet_train_%j.err
#SBATCH --mail-user=gianluca.caregnato@studenti.unipd.it
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=08:00:00
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -x
set -e

cd /home/caregnatog/VisionExplainability/CNN

# Use official PyTorch CUDA image through Singularity
IMG="docker://pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime"

REQ="/home/caregnatog/VisionExplainability/requirements.txt"

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"

# 1) Install packages into ~/.local (PERSISTENT)
srun singularity exec --nv "$IMG" \
    bash -lc "python3 -m pip install --user --upgrade pip && python3 -m pip install --user -r '$REQ'"

# 2) Sanity check inside container
srun singularity exec --nv "$IMG" \
    bash -lc "python3 -c 'import numpy, pytorch_grad_cam; print(\"numpy OK:\", numpy.__version__)'"

# 3) Run your script
srun singularity exec --nv "$IMG" \
    bash -lc 'python3 /home/caregnatog/VisionExplainability/CNN/explainability.py'

echo "Job finished at: $(date)"
