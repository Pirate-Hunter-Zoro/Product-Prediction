#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=c3_accel
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_logs/evaluation_out.txt
#SBATCH -e slurm_logs/evaluation_err.txt
#SBATCH --gres=gpu:1

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=amazon-m2-venv
N_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10
export PYTHONNOUSERSITE=1

mkdir -p slurm_logs

# Explicitly pass the memory limit to srun to resolve the conflict
which python
python -c "import recbole; print(recbole.__file__)"
python scripts/evaluate.py