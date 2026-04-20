#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=c3_accel
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_logs/evaluation_%x_out.txt
#SBATCH -e slurm_logs/evaluation_%x_err.txt
#SBATCH --gres=gpu:1

# Exit immediately if command fails
set -e

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Unspecified Model...." >&2
    exit 2
fi
shift # Consume the first model argument
EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
    # More arguments
    EXTRA_ARGS="--extra-metrics $@"
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
python scripts/evaluate.py --model "$MODEL" $EXTRA_ARGS # Note that $EXTRA_ARGS is unquoted due to the fact that it should be potentially multiple arguments