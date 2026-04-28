#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_logs/training_%x_out.txt
#SBATCH -e slurm_logs/training_%x_err.txt
#SBATCH --gres=gpu:1

# Exit immediately if any command fails
set -e

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Unspecified Model...." >&2
    exit 2
fi

shift # Renumber $2...$N to $1...$(N-1) by popping off the old $1
SLOTS_ARGS=""
if [ "$#" -gt 0 ]; then
    SLOTS_ARGS="--slots $@" # Expand slots to all remaining arguments
fi

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
python scripts/train.py --model "$MODEL" $SLOTS_ARGS # Unquoted slots args for argument splitting