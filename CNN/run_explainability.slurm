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
#SBATCH --mem=16G

set -x  # print commands

# 1) Go to project
cd /home/caregnatog/VisionExplainability/CNN

# 2) Activate venv
source /home/caregnatog/venvs/VisionExpainability/bin/activate

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Node list: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "PATH: $PATH"
echo "Python in use: $(which python3)"

# 3) Show Python version and site-packages
python3 --version
python3 -c "import sys, site; print(sys.executable); print(sys.version); print('site-packages:', site.getsitepackages())"

# 4) Check that numpy and pytorch_grad_cam are importable
python3 -c "import numpy; print('numpy OK:', numpy.__version__)"
python3 -c 'import pytorch_grad_cam; print("pytorch_grad_cam OK")'

# 5) Run your script directly with venv python (NO Singularity)
srun python3 explainability.py

echo "Job finished at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"