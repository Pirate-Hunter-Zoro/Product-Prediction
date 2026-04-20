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
PYTHONNOUSERSITE=1 python -m pip install --no-user recbole "ray[tune]" kmeans-pytorch "numpy<1.24.0" 
PYTHONNOUSERSITE=1 python -m pip install --no-user matplotlib
PYTHONNOUSERSITE=1 python -c "import matplotlib; import recbole; import setuptools; import torch; from recbole.quick_start import run_recbole; print(recbole.__file__); print(setuptools.__version__)"
PYTHONNOUSERSITE=1 python -c "import matplotlib.pyplot as plt; plt.figure(); print('ok')"