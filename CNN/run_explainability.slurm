#!/bin/bash -l

#SBATCH --job-name=resnet_train
#SBATCH --output=resnet_train_%j.out
#SBATCH --error=resnet_train_%j.err

#SBATCH --time=08:00:00
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -x

cd "$HOME/VisionExplainability/CNN"

# Scratch directory on /ext
SCRATCH=/ext/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
export TMP="$SCRATCH"
export TEMP="$SCRATCH"
echo "Scratch dir: $SCRATCH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun singularity exec --nv /nfsd/opt/sif-images/cv-ml-torch.sif \
    bash -lc "source $HOME/venvs/gradcam_env/bin/activate && \
              cd $HOME/VisionExplainability/CNN && \
              python explainability.py"

echo "Job finished at: $(date)"