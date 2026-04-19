#!/bin/bash

# Exit immediately if any command fails
set -e

sbatch --job-name=GRU4Rec run_training.sbatch GRU4Rec
sbatch --job-name=NARM run_training.sbatch NARM