# Amazon-M2 Next Product Recommendation

Session-based next product recommendation on the Amazon-M2 multilingual shopping dataset. Built with [RecBole](https://recbole.io/) for a graduate course final project.

## Task

Given a shopping session (a sequence of viewed/purchased products), predict the next product the user will interact with. Evaluation uses MRR@100, Recall@100, and NDCG@100.

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
  preprocess.py            Data preprocessing (CSV -> RecBole .inter, SEED=42)
  config.yaml              RecBole configuration
  train.py                 Training entrypoint
  evaluate.py              Standalone evaluation of saved checkpoints
  locale_map.py            Builds session_id -> locale lookup (parquet)
  split_test_by_locale.py  Splits test.inter into per-locale test_<locale>.inter files
  pop_baseline.py          Hand-rolled Popularity baseline (global + session-aware)
create_env.sh              Conda environment setup (cluster)
run_preprocessing.sbatch   Slurm job for preprocessing (CPU only)
run_training.sbatch        Slurm job for training (takes model name as $1)
run_evaluation.sbatch      Slurm job for evaluation (takes model name as $1)
run_pop_evaluation.sbatch  Slurm job for pop_baseline.py (CPU only, ~1 min)
submit_training.sh         Wrapper: submits GRU4Rec, NARM, and Pop training jobs
submit_evaluation.sh       Wrapper: submits GRU4Rec, NARM, and Pop evaluation jobs
data_investigation.ipynb   Initial data exploration
data/
  sessions_train.csv       Raw session data (gitignored)
  products_train.csv       Raw product metadata (gitignored)
  amazon_m2/               Generated RecBole artifacts:
                             amazon_m2.{train,valid,test}.inter (preprocess.py)
                             amazon_m2.test_<locale>.inter      (split_test_by_locale.py)
                             locale_map.parquet                 (locale_map.py)
saved/                     Model checkpoints
slurm_logs/                Slurm stdout/stderr logs
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

The end-to-end cluster pipeline has three Slurm stages: preprocessing, training, and evaluation. Environment setup is covered above. Preprocessing runs once and writes `.inter` files that training and evaluation both consume, so it must complete before training jobs are submitted.

### Preprocessing

Convert raw CSV data to RecBole `.inter` format with a seeded 80/10/10 train/valid/test split:

```bash
sbatch run_preprocessing.sbatch
```

Output lands in `data/amazon_m2/`. The shuffle is seeded (`SEED=42` in `scripts/preprocess.py`) so the split is reproducible from a clean clone. No GPU is requested.

For local smoke tests on a small subset, call the script directly:

```bash
python scripts/preprocess.py --nrows 3000
```

### Training

Submit GRU4Rec, NARM, and Pop training jobs concurrently:

```bash
bash submit_training.sh
```

This dispatches three Slurm jobs via `run_training.sbatch`, each tagged with a distinct `--job-name` so their logs do not collide. `run_training.sbatch` requires the model name as `$1` and exits with code 2 if it is missing; preprocessing is no longer part of this sbatch, so concurrent training jobs cannot race on the `.inter` files.

To train a single model by hand:

```bash
sbatch --job-name=<ModelName> run_training.sbatch <ModelName>
```

Logs are written to `slurm_logs/training_<JobName>_out.txt` and `slurm_logs/training_<JobName>_err.txt`.

### Evaluation

Evaluate the most recent GRU4Rec, NARM, and Pop checkpoints on the test set:

```bash
bash submit_evaluation.sh
```

`run_evaluation.sbatch` takes the model name as `$1` and forwards it to `scripts/evaluate.py --model <ModelName>`. The evaluator globs `saved/<ModelName>*.pth`, selects the newest file by modification time, and prints MRR@100, Recall@100, and NDCG@100 as JSON to stdout.

To evaluate a single model by hand:

```bash
sbatch --job-name=<ModelName> run_evaluation.sbatch <ModelName>
```

Logs are written to `slurm_logs/evaluation_<JobName>_out.txt` and `slurm_logs/evaluation_<JobName>_err.txt`.

#### Pop baseline (special case)

RecBole's built-in `Pop` model is a `GeneralRecommender` and is incompatible with this project's sequential configuration (`benchmark_filename`, `alias_of_item_id`, session_id as `USER_ID_FIELD`). Training "succeeds" but produces garbage popularity counts (~0.0002 MRR). Pop is therefore evaluated through a hand-rolled script that reads the `.inter` files directly:

```bash
sbatch run_pop_evaluation.sbatch
```

This runs `scripts/pop_baseline.py` on CPU, completes in roughly one minute, and prints two JSON blocks to stdout — one for global-Pop ranking, one for session-aware Pop. Session-aware is reported as a negative-result baseline: on Task 1 data the `next_item` is disjoint from `prev_items` on every row by construction, so session-aware Pop cannot outperform global Pop. Both numbers appear in the final results table.

## Models

| Model   | Type     | Status    |
| ------- | -------- | --------- |
| GRU4Rec | Baseline | Trained, evaluated |
| NARM    | Baseline | Trained, evaluated |
| Pop     | Baseline | Evaluated (hand-rolled, see `scripts/pop_baseline.py`) |
| TBD     | Novel    | In design |

## Known Issues

- **PyTorch 2.6 / RecBole incompatibility:** PyTorch 2.6 changed the default of `torch.load` to `weights_only=True`, which breaks RecBole's checkpoint loading. Both `train.py` and `evaluate.py` include a monkey-patch that forces `weights_only=False`.
- **RecBole 1.2.0 requires numpy < 1.24.0** due to use of the removed `np.float_` alias.
- **RecBole undeclared dependencies:** ray, kmeans-pytorch, and setuptools (for `pkg_resources`) must be installed manually.
- **conda activation on the cluster** requires `module purge && module load Anaconda3/2025.06-0` before any `conda activate` call. Without this, a different conda installation may be used, pointing to a different environment with the same name.
- **`~/.local` site-packages interference:** The cluster's shared filesystem causes pip to fall back to `--user` installs. All pip commands in `create_env.sh` use `--no-user`, and `run_training.sbatch` sets `PYTHONNOUSERSITE=1` at runtime.
- **Full dataset requires GPU cluster.** The 3.6M sessions and 1.4M items will exhaust memory on a laptop. Use `--nrows` for local testing.
