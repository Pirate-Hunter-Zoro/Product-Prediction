#!/bin/bash
#SBATCH --job-name=encode_titles
#SBATCH --partition=c3_accel
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm_logs/%x_out.txt
#SBATCH -e slurm_logs/%x_err.txt
#SBATCH --gres=gpu:1

# Exit immediately if any command fails
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

which python
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
python scripts/encode_titles.py --batch-size 256