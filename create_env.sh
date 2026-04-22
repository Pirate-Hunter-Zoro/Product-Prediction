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
PYTHONNOUSERSITE=1 python -m pip install --no-user sentence-transformers "numpy<1.24.0"
PYTHONNOUSERSITE=1 python -m pip install --no-user matplotlib
PYTHONNOUSERSITE=1 python -c "import matplotlib; import recbole; import setuptools; import torch; import numpy; import sentence_transformers; from recbole.quick_start import run_recbole; print(recbole.__file__); print(setuptools.__version__); print('numpy', numpy.__version__)"
PYTHONNOUSERSITE=1 python -c "import matplotlib.pyplot as plt; plt.figure(); print('ok')"
PYTHONNOUSERSITE=1 python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('/media/studies/ehr_study/analysis/mferguson/models/paraphrase-multilingual-MiniLM-L12-v2'); v = m.encode(['hello', 'bonjour'], normalize_embeddings=True); print('st', v.shape, v.dtype)"