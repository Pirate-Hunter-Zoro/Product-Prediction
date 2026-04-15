#!/bin/bash
set -e
module purge
module load Anaconda3/2025.06-0                            
eval "$(conda shell.bash hook)" 
conda env remove -n amazon-m2-venv -y || true                                                                                     
conda create -n amazon-m2-venv python=3.10 -y
conda activate amazon-m2-venv
PYTHONNOUSERSITE=1 python -m pip install --no-user torch --index-url https://download.pytorch.org/whl/cu118
PYTHONNOUSERSITE=1 python -m pip install --no-user "setuptools<80"
PYTHONNOUSERSITE=1 python -m pip install --no-user recbole "ray[tune]" kmeans-pytorch "numpy<2.0" 
PYTHONNOUSERSITE=1 python -c "import recbole; import setuptools; import torch; from recbole.quick_start import run_recbole; print(recbole.__file__); print(setuptools.__version__)"