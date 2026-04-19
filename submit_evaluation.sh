#!/bin/bash

# Exit immediately if any command fails
set -e

sbatch --job-name=GRU4Rec run_evaluation.sbatch GRU4Rec
sbatch --job-name=NARM run_evaluation.sbatch NARM