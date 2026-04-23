import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def load_train_vocab(inter_dir: Path) -> set[str]:
    """Return every item appearing in the training set as either part of an input item sequence or a target item

    Args:
        inter_dir (Path): Directory containing amazon_m2.train.inter

    Returns:
        set[str]: All relevant item ids present
    """
    train_path = inter_dir / "amazon_m2.train.inter"
    train_df = pd.read_csv(train_path, sep="\t")
    target_series = train_df["item_id:token"].astype(str)
    # Yield one row of - a list of item IDs for each transaction's input sequence of items
    input_series = train_df["item_id_list:token_seq"].str.split()
    # Now turn that into each ITEM getting its own row - flattening all the lists
    input_series = input_series.explode()
    # Join all of the target ids with all the ids from input sequences of items
    joint_series = pd.concat([target_series, input_series])
    # Filter out the items which are empty or isna for some reason
    joint_series = joint_series[joint_series.notna() & (joint_series != "")]
    return set(joint_series)

def load_train_prices(attributes_path: Path, train_vocab: set[str]) -> np.ndarray:
    """Return array of prices restricted to items appearing in the input training vocab

    Args:
        attributes_path (Path): Full path to data/amazon_m2/item_attributes.parquet
        train_vocab (set[str]): Set returned by load_train_vocab

    Returns:
        np.ndarray: Type float64, 1-D (n_train_items,)
    """
    attributes_df = pd.read_parquet(attributes_path)
    # We only care about item_id and price
    attributes_df = attributes_df[['item_id', 'price']]
    # Filter to only items present in the training vocab
    train_rows = attributes_df[attributes_df['item_id'].isin(train_vocab)]
    prices = train_rows["price"].to_numpy()
    assert prices.dtype == np.float64, f"Expected prices data type of {np.float64} but found {prices.dtype}"
    return prices

def compute_quantile_boundaries(train_prices: np.ndarray, n_bins: int) -> np.ndarray:
    """Produce cut points to create equal frequency bins of prices

    Args:
        train_prices (np.ndarray): Output of load_train_prices
        n_bins (int): Number of bins

    Returns:
        np.ndarray: Shape (n_bins - 1,), the values at the cut points of quantile levels 1/n_bins, 2/n_bins, ...(n_bins-1)/n_bins (monotonic non-decreasing)
    """
    if n_bins < 2:
        raise ValueError(f"Expected number of bins at least 2, but received {n_bins}")
    quantiles = np.linspace(1 / n_bins, (n_bins-1) / n_bins, n_bins-1)
    # Find all the values within the prices array that correspond with each quantile position
    return np.quantile(train_prices, quantiles)

def bucketize_all_prices(
    attributes_path: Path,
    boundaries: np.ndarray,
    n_bins: int
) -> tuple[list[str], np.ndarray]:
    """For every item in the attributes parquet, assign a bin_idx based on its value

    Args:
        attributes_path (Path): Full path to item_attributes.parquet
        boundaries (np.ndarray): Output of compute_quantile_boundaries
        n_bins (int): Bin count (from upstream)

    Returns:
        tuple[list[str], np.ndarray]: Item ids, respective bin indices
    """
    attributes_df = pd.read_parquet(attributes_path)
    relevant_attributes = attributes_df[['item_id', 'price']]
    item_ids = relevant_attributes['item_id'].tolist()
    prices = relevant_attributes['price'].to_numpy()
    assert prices.dtype == np.float64, f"Expected prices to have type {np.float64}, but found {prices.dtype}"
    # Find bin_indices such that if x lands at index i, boundaries[i-1] <= x < boundaries[i]
    bin_indices = np.digitize(prices, boundaries).astype(np.int64)
    assert bin_indices.min() >= 0 and bin_indices.max() <= n_bins - 1, f"Boundaries had shape {boundaries.shape} which is incompatible with {n_bins} bins"
    return (item_ids, bin_indices)

def save_outputs(
    item_ids: list[str],
    bin_indices: np.ndarray,
    boundaries: np.ndarray,
    bins_output: Path,
    boundaries_output: Path,
):
    """Persist binning results to disk

    Args:
        item_ids (list[str]): List returned by bucketize_all_prices
        bin_indices (np.ndarray): Aligned with item IDs by position
        boundaries (np.ndarray): Cut points from compute_quantile_boundaries
        bins_output (Path): Destination for price_bins.pt
        boundaries_output (Path): Destination for price_boundaries.pt
    """
    bin_tensor = torch.from_numpy(bin_indices)
    assert bin_tensor.dtype == torch.int64, f"Expected the bin_tensor to have type {torch.int64} but yielded {bin_tensor.dtype} from numpy array of type {bin_indices.dtype}"
    boundaries_tensor = torch.from_numpy(boundaries.astype(np.float32))
    payload = {
        "item_ids": item_ids,
        "bin_idx": bin_tensor
    }
    bins_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, bins_output)
    boundaries_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(boundaries_tensor, boundaries_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-dir", type=Path, default=Path("data/amazon_m2"))
    parser.add_argument("--attributes", type=Path, default=Path("data/amazon_m2/item_attributes.parquet"))
    parser.add_argument("--output-bins", type=Path, default=Path("data/amazon_m2/price_bins.pt"))
    parser.add_argument("--output-boundaries", type=Path, default=Path("data/amazon_m2/price_boundaries.pt"))
    parser.add_argument("--n-bins", type=int, default=32)
    args = parser.parse_args()
    train_vocab = load_train_vocab(args.inter_dir)
    print(f"Vocab size: {len(train_vocab)}", flush=True)
    train_prices = load_train_prices(args.attributes, train_vocab)
    print(f"Train prices shape: {train_prices.shape}, Train prices type: {train_prices.dtype}", flush=True)
    boundaries = compute_quantile_boundaries(train_prices, args.n_bins)
    print(f"Boundaries shape: {boundaries.shape}, Min: {boundaries.min():.4f}, Median: {np.median(boundaries):.4f}, Max: {boundaries.max():.4f}", flush=True)
    item_ids, bin_indices = bucketize_all_prices(args.attributes, boundaries, args.n_bins)
    print(f"Bin indices shape: {bin_indices.shape}, Min: {bin_indices.min()}, Max: {bin_indices.max()}", flush=True)
    # Observe the lowest and highest bin counts keeping in mind what the average bin count must be
    counts = np.bincount(bin_indices, minlength=args.n_bins)
    print(f"Minimum bin count: {counts.min()}, Maximum bin count: {counts.max()}", flush=True)
    save_outputs(item_ids, bin_indices, boundaries, args.output_bins, args.output_boundaries)
    print(f"Wrote {args.output_bins} and {args.output_boundaries}", flush=True)