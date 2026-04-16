# Amazon-M2 Next Product Recommendation

Session-based next product recommendation on the Amazon-M2 multilingual shopping dataset. Built with [RecBole](https://recbole.io/) for a graduate course final project.

## Task

Given a shopping session (a sequence of viewed/purchased products), predict the next product the user will interact with. Evaluation uses MRR@100 and Recall@100.

## Dataset

The Amazon-M2 dataset contains approximately 3.6M training sessions across 1.4M products in 6 locales.

| Locale | Sessions |
| ------ | -------- |
| UK     | 1.18M    |
| DE     | 1.11M    |
| JP     | 979K     |
| IT     | 127K     |
| FR     | 118K     |
| ES     | 89K      |

Sessions average approximately 4 items (median 3, max 474).

The raw data files (`data/sessions_train.csv`, `data/products_train.csv`) are gitignored and must be obtained separately, then placed in `data/`.

## Project Structure

```text
scripts/
  preprocess.py       Data preprocessing (CSV -> RecBole .inter format)
  config.yaml         RecBole configuration
  train.py            Training entrypoint
  evaluate.py         Standalone evaluation of saved checkpoints
create_env.sh         Conda environment setup (cluster)
run_training.sbatch   Slurm job script for GPU cluster training
data_investigation.ipynb  Initial data exploration
data/
  sessions_train.csv  Raw session data (gitignored)
  products_train.csv  Raw product metadata (gitignored)
  amazon_m2/          Generated .inter files (output of preprocess.py)
saved/                Model checkpoints
slurm_logs/           Slurm stdout/stderr logs
```

## Environment Setup

### Cluster (training)

Run `create_env.sh` on the GPU cluster. This script handles conda environment creation and installs all dependencies (PyTorch with CUDA 11.8, RecBole, ray, kmeans-pytorch, numpy). It must be run after loading the correct Anaconda module:

```bash
module purge && module load Anaconda3/2025.06-0
bash create_env.sh
```

Key dependencies:

- Python 3.10
- PyTorch (CUDA 11.8)
- RecBole 1.2.0
- numpy < 2.0
- ray[tune], kmeans-pytorch, setuptools

### VS Code (development)

If VS Code cannot resolve RecBole imports despite having the correct interpreter selected, add the environment's site-packages to Pylance's search paths. This is necessary because the conda environment lives on a network filesystem that Pylance does not automatically index.

In `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "/media/studies/ehr_study/analysis/mferguson/venvs/amazon-m2-venv/lib/python3.10/site-packages"
    ]
}
```

Set the Python interpreter to the full path of the environment's binary:

```text
/media/studies/ehr_study/analysis/mferguson/venvs/amazon-m2-venv/bin/python
```

## Usage

### Preprocessing

Convert raw CSV data to RecBole `.inter` format with an 80/10/10 train/valid/test split:

```bash
python scripts/preprocess.py
```

For local smoke tests on a small subset:

```bash
python scripts/preprocess.py --nrows 3000
```

### Training

Submit a training job on the cluster via Slurm:

```bash
sbatch run_training.sbatch
```

The sbatch script runs preprocessing followed by GRU4Rec training. To train a different model, change the `--model` argument in the sbatch script (e.g., `--model NARM`).

Logs are written to `slurm_logs/training_out.txt` and `slurm_logs/training_err.txt`.

### Evaluation

Evaluate a saved checkpoint on the test set without retraining:

```bash
python scripts/evaluate.py --model_file saved/GRU4Rec-Apr-15-2026_14-14-09.pth
```

## Models

| Model   | Type     | Status    |
| ------- | -------- | --------- |
| GRU4Rec | Baseline | Trained   |
| NARM    | Baseline | Planned   |
| TBD     | Novel    | In design |

## Known Issues

- **PyTorch 2.6 / RecBole incompatibility:** PyTorch 2.6 changed the default of `torch.load` to `weights_only=True`, which breaks RecBole's checkpoint loading. Both `train.py` and `evaluate.py` include a monkey-patch that forces `weights_only=False`.
- **RecBole 1.2.0 requires numpy < 1.24.0** due to use of the removed `np.float_` alias.
- **RecBole undeclared dependencies:** ray, kmeans-pytorch, and setuptools (for `pkg_resources`) must be installed manually.
- **conda activation on the cluster** requires `module purge && module load Anaconda3/2025.06-0` before any `conda activate` call. Without this, a different conda installation may be used, pointing to a different environment with the same name.
- **`~/.local` site-packages interference:** The cluster's shared filesystem causes pip to fall back to `--user` installs. All pip commands in `create_env.sh` use `--no-user`, and `run_training.sbatch` sets `PYTHONNOUSERSITE=1` at runtime.
- **Full dataset requires GPU cluster.** The 3.6M sessions and 1.4M items will exhaust memory on a laptop. Use `--nrows` for local testing.
