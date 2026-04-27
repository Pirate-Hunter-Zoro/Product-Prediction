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
  train.py                 Training entrypoint. Accepts --model <name>; monkey-patches recbole.quick_start.quick_start.get_model so "NovelModel" resolves to the local NovelModel class (RecBole's get_model uses importlib find_spec on a hypothetical filename and cannot resolve runtime-only classes), while all other names fall through to the original get_model unchanged. torch.load is also monkey-patched to weights_only=False for RecBole 1.2.0 compatibility.
  evaluate.py              Standalone evaluation of saved checkpoints
  locale_map.py            Builds session_id -> locale lookup (parquet)
  split_test_by_locale.py  Splits test.inter into per-locale test_<locale>.inter files
  pop_baseline.py          Hand-rolled Popularity baseline (global + session-aware)
  build_item_attributes.py Builds per-item attribute parquet (item_id, title, brand, price, color) from products_train.csv filtered to the RecBole vocab
  encode_text_attribute.py Encodes any text column from item_attributes.parquet with paraphrase-multilingual-MiniLM-L12-v2 (selected by --column {title,brand,color}); writes aligned {item_ids, embeddings} tensor pickle to data/amazon_m2/{column}_embeddings.pt
  bucketize_price.py       Bucketizes per-item price into 32 quantile bins (boundaries from TRAIN prices only); writes data/amazon_m2/price_bins.pt ({item_ids, bin_idx}) and data/amazon_m2/price_boundaries.pt (bare FloatTensor[31])
  attribute_loader.py      Model-side helpers that load the precomputed attribute artifacts into tensors indexed by RecBole internal item ID. load_text_embedding reads a {title,brand,color}_embeddings.pt pickle and returns FloatTensor[num_items, 384] with row 0 (PAD) zeroed. load_price_bins reads price_bins.pt and returns LongTensor[num_items] with row 0 = n_bins (reserved PAD sentinel) and rows 1..num_items-1 = bin index in [0, n_bins-1]. Both use vectorized dataset.token2id; neither is a CLI entry point. Both filter ext_ids to those present in dataset.field2token_id[item_id_field] before calling token2id (which raises ValueError on unknown tokens), and print a flushing "Missing N out of M items in the input dataset..." line when any items are dropped - silent on full-data runs, expected to fire on smoke subsets where the cached embeddings cover a larger vocab than the current .inter split.
  novel_model.py           NovelModel(SequentialRecommender) subclass for the novel cross-attention method. __init__ is fully allocated: four register_buffer calls (title/brand/color text embeddings plus price_bin_idx), learnable item_embedding (padding_idx=0), emb_dropout, GRU (batch_first=True), three separate text projections (title_proj, brand_proj, color_proj) from 384 to hidden_size, price_embedding (nn.Embedding(n_price_bins + 1, hidden_size)), and cross_attn (nn.MultiheadAttention with batch_first=True). All four RecBole-required method bodies are implemented: forward returns FloatTensor[batch, hidden_size] (GRU session state fused with cross-attended item-attribute bag); calculate_loss scores fused_output against the full item_embedding.weight table and returns F.cross_entropy(logits, target); predict returns row-wise dot product (fused * item_embedding(ITEM_ID)).sum(dim=1) of shape (batch,) for sampled-candidate eval and RecBole's get_flops probe at startup (uses self.ITEM_ID, NOT self.POS_ITEM_ID); full_sort_predict performs the same matmul against item_embedding.weight.T and returns raw FloatTensor[batch, num_items] scores for RecBole's eval pipeline. All three scoring methods use the identical dot-product rule, so get_flops, sampled-candidate eval, and full-sort eval cannot disagree. Config keys consumed (all wired into scripts/config.yaml): TITLE_EMBEDDING_PATH, BRAND_EMBEDDING_PATH, COLOR_EMBEDDING_PATH, PRICE_BINS_PATH, n_price_bins, num_layers, dropout_prob, num_heads, hidden_size, loss_type.
create_env.sh              Conda environment setup (cluster)
run_preprocessing.sbatch   Slurm job for preprocessing (CPU only)
run_training.sbatch        Slurm job for training (takes model name as $1)
run_evaluation.sbatch      Slurm job for evaluation (takes model name as $1)
run_pop_evaluation.sbatch  Slurm job for pop_baseline.py (CPU only, ~1 min)
run_encode_text_attributes.sbatch  Slurm job for encode_text_attribute.py (takes column name as $1; 1 GPU, ~2 min at batch 256)
submit_training.sh         Wrapper: submits GRU4Rec, NARM, Pop, and NovelModel training jobs
submit_evaluation.sh       Wrapper: submits GRU4Rec, NARM, NovelModel, and Pop evaluation jobs
submit_embedding.sh        Wrapper: submits title, brand, and color encoding jobs with distinct --job-name values
data_investigation.ipynb   Initial data exploration
data/
  sessions_train.csv       Raw session data (gitignored)
  products_train.csv       Raw product metadata (gitignored)
  amazon_m2/               Generated RecBole artifacts:
                             amazon_m2.{train,valid,test}.inter (preprocess.py)
                             amazon_m2.test_<locale>.inter      (split_test_by_locale.py)
                             locale_map.parquet                 (locale_map.py)
                             item_attributes.parquet            (build_item_attributes.py)
                             title_embeddings.pt                (encode_text_attribute.py --column title)
                             brand_embeddings.pt                (encode_text_attribute.py --column brand)
                             color_embeddings.pt                (encode_text_attribute.py --column color)
                             price_bins.pt                      (bucketize_price.py)
                             price_boundaries.pt                (bucketize_price.py)
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
- numpy < 1.24.0
- ray[tune], kmeans-pytorch, setuptools < 80, matplotlib
- sentence-transformers (for `paraphrase-multilingual-MiniLM-L12-v2` title embeddings)

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

#### Scope

Primary results are reported across all six locales (UK, DE, JP, IT, FR, ES) unioned into a single "Overall" number, plus one row per locale. A secondary "Overall (UK/DE/JP)" row is emitted alongside as a session-count-weighted average of the UK, DE, and JP per-locale rows (denominator = 327,319 test sessions) to enable direct comparison to Amazon-M2 paper Table 3, which evaluates Task 1 on the three large locales only. Because MRR@100, Recall@100, and NDCG@100 are all row-wise means, the weighted average is exact and requires no second evaluation pass. The two "Overall" numbers live in different universes and must always be labeled with their scope when cited.

Evaluate the most recent GRU4Rec, NARM, and Pop checkpoints on the test set:

```bash
bash submit_evaluation.sh
```

`run_evaluation.sbatch` takes the model name as `$1` and zero or more extra metric names as `$2..$N`, forwarding the extras to `scripts/evaluate.py` as `--extra-metrics <names>`. The evaluator globs `saved/<ModelName>*.pth`, selects the newest file by modification time, unions any `--extra-metrics` into the checkpoint's stored `config["metrics"]` before instantiating the trainer, iterates the six per-locale `amazon_m2.test_<locale>.inter` files, and prints a single JSON dict to stdout with eight keys: `UK`, `DE`, `JP`, `IT`, `FR`, `ES`, `Overall` (the full-test run), and `Overall (UK/DE/JP)` (the paper-parity weighted average of the three large locales, computed inline from the per-locale rows). The per-locale loop reuses the model and dataset vocab loaded once at the top — each locale builds a filtered `FullSortEvalDataLoader` by masking `test_data._dataset.inter_feat` on `session_id` via `dataset.token2id`, without retraining or reloading the checkpoint.

The extras mechanism exists so that NDCG@100 can be computed on GRU4Rec and NARM checkpoints that were trained before NDCG was added to `config.yaml`, without retraining. `submit_evaluation.sh` appends `NDCG` after the model name for both deep baselines. Pop is evaluated through a separate hand-rolled script and already reports NDCG directly; it does not participate in the per-locale loop.

To evaluate a single model by hand (without extras):

```bash
sbatch --job-name=<ModelName> run_evaluation.sbatch <ModelName>
```

To evaluate a single model by hand with NDCG@100 backfilled:

```bash
sbatch --job-name=<ModelName> run_evaluation.sbatch <ModelName> NDCG
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
| Cross-attention over item attributes | Novel    | Smoke-tested (item-attribute parquet built; title/brand/color embeddings cached via `scripts/encode_text_attribute.py`; price bucketized into 32 quantile bins via `scripts/bucketize_price.py`; model-side loader helpers in `scripts/attribute_loader.py` with vocab-tolerance filter and flushing `Missing N out of M items` alarm; `SequentialRecommender` subclass in `scripts/novel_model.py` with `__init__`, `forward`, `calculate_loss`, `predict`, and `full_sort_predict` all implemented; config keys wired into `scripts/config.yaml` (TITLE/BRAND/COLOR/PRICE paths, n_price_bins, num_layers, dropout_prob, num_heads, hidden_size, loss_type); class registered into `scripts/train.py` via a monkey-patch on `recbole.quick_start.quick_start.get_model`; end-to-end smoke test passed on the 3000-row split 2026-04-27 (constructor → forward → calculate_loss → predict → full_sort_predict, exit 0, garbage metrics expected at that data scale); full-data .inter restore submitted as job 1873177; full-scale training pending preprocess completion) |

## Known Issues

- **PyTorch 2.6 / RecBole incompatibility:** PyTorch 2.6 changed the default of `torch.load` to `weights_only=True`, which breaks RecBole's checkpoint loading. Both `train.py` and `evaluate.py` include a monkey-patch that forces `weights_only=False`.
- **RecBole 1.2.0 requires numpy < 1.24.0** due to use of the removed `np.float_` alias.
- **RecBole undeclared dependencies:** ray, kmeans-pytorch, and setuptools (for `pkg_resources`) must be installed manually.
- **conda activation on the cluster** requires `module purge && module load Anaconda3/2025.06-0` before any `conda activate` call. Without this, a different conda installation may be used, pointing to a different environment with the same name.
- **`~/.local` site-packages interference:** The cluster's shared filesystem causes pip to fall back to `--user` installs. All pip commands in `create_env.sh` use `--no-user`, and `run_training.sbatch` sets `PYTHONNOUSERSITE=1` at runtime.
- **Full dataset requires GPU cluster.** The 3.6M sessions and 1.4M items will exhaust memory on a laptop. Use `--nrows` for local testing.
