#!/bin/bash
set -e
module purge
module load Anaconda3/2025.06-0                            
eval "$(conda shell.bash hook)" 
conda env remove -n amazon-m2-venv -y || true                                                                                     
conda create -n amazon-m2-venv python=3.10 -y
conda activate amazon-m2-venv
python -m pip install --no-user torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-user setuptools
python -m pip install --no-user recbole "ray[tune]" kmeans-pytorch "numpy<2.0" 
PYTHONNOUSERSITE=1 python -c "import recbole; import torch; print(recbole.__file__)"