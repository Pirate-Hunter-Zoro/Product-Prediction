#!/bin/bash

# Exit immediately if any command fails
set -e

sbatch --job-name=embed_title run_encode_text_attributes.sbatch title
sbatch --job-name=embed_brand run_encode_text_attributes.sbatch brand
sbatch --job-name=embed_color run_encode_text_attributes.sbatch color