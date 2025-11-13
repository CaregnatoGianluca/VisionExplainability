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
# Number of tasks. For a single GPU job, you typically only need 1 task.
# The training is not intrinsically multi-threaded/distributed here.
#SBATCH --ntasks 1

# Number of CPUs per task. Use this to allocate CPU cores for data loading/preprocessing.
# Your code uses DATASET_WORK_NUMBER = 8, so requesting 8 CPUs is appropriate.
#SBATCH --cpus-per-task 8

#SBATCH --mem 16G

# --- Execute Commands ---

set -x  # print commands as they run

cd $HOME/VisionExplainability/CNN

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Node list: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "PATH: $PATH"

source "$HOME/venvs/VisionExplainability/bin/activate"

echo "Python in use: $(which python3)"
python3 --version

srun python3 explainability.py

# srun singularity exec --nv /nfsd/opt/sif-images/tensorflow_latest-gpu-new.sif python ResNet_training.py

# Optional: Print out the final allocation for verification
echo "Job finished at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"