#!/bin/bash -l

#SBATCH --job-name resnet_train

#SBATCH --output resnet_train_%j.out
#SBATCH --error resnet_train_%j.err

#SBATCH --mail-user gianluca.caregnato@studenti.unipd.it
#SBATCH --mail-type BEGIN,END,FAIL

#SBATCH --time 08:00:00

#SBATCH --partition allgroups

# --- GPU Configuration ---
#SBATCH --gres=gpu:a40:1

# --- CPU/Memory Configuration ---
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G

set -x  # print commands as they run

# 1) Go to the repo (fix the spelling of the folder name!)
cd $HOME/VisionExplainability/CNN

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Node list: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "PATH: $PATH"

# Optional: match OpenMP threads to SLURM CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2) Use YOUR SIF instead of the shared tensorflow image
#    Adjust the path to gradcam.sif if you stored it elsewhere.
srun singularity exec --nv $HOME/sif/gradcam.sif \
     python ResNet_training.py

echo "Job finished at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
