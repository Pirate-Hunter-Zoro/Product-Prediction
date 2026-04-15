#!/bin/bash
set -e
module purge
module load Anaconda3/2025.06-0                            
eval "$(conda shell.bash hook)" 
conda env remove -n amazon-m2-venv -y || true                                                                                     
conda create -n amazon-m2-venv python=3.10 -y
conda activate amazon-m2-venv
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install setuptools
pip install recbole "ray[tune]" kmeans-pytorch "numpy<2.0" 
python -c "import recbole; import torch; print('ok', recbole.__file__, torch.__version__)"