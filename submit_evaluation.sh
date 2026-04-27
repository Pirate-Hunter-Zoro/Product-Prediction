#!/bin/bash

# Exit immediately if any command fails
set -e

sbatch --job-name=GRU4Rec run_evaluation.sbatch GRU4Rec NDCG
sbatch --job-name=NARM run_evaluation.sbatch NARM NDCG
sbatch --job-name=NovelModel run_evaluation.sbatch NovelModel
sbatch run_pop_evaluation.sbatch