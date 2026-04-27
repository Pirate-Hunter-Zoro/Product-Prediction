import torch
from pathlib import Path

# NOTE - by RecBole convention, a Row 0 = PAD value is always present, so num_items should be thought of as (number of real items) + 1

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = _patched_load

def load_text_embedding(
    pickle_path: str | Path,
    dataset,
    item_id_field: str = "item_id",
) -> torch.Tensor:
    """Load attribute embeddings for all items

    Args:
        pickle_path (str | Path): Path to whatever {column}_embeddings.pt embeddings specified
        dataset (RecBole Dataset or SequentialDataset): Underlying dataset
        item_id_field (str, optional): RecBole field name holding the item IDs. Defaults to "item_id".

    Returns:
        torch.Tensor: Shape (num_items, embedding_dim); Row 0 is all zeros while rows i > 0 hold attribute embeddings for items of corresponding id=i
    """
    # Load item ALL IDS and see which ones are present in the given dataset
    pickle_path = Path(pickle_path) # In case of str
    payload = torch.load(pickle_path)
    ext_ids, embeddings = payload['item_ids'], payload['embeddings']
    known_tokens = set(dataset.field2token_id[item_id_field])
    present = [1 if id in known_tokens else 0 for id in ext_ids]
    n_total = len(ext_ids)
    n_missing = n_total - sum(present)
    if n_missing > 0:
        print(f"Missing {n_missing} out of {n_total} items in the input dataset...", flush=True)
    ext_ids = [ext_ids[i] for i in range(n_total) if present[i]]
    embeddings = embeddings[torch.as_tensor(present, dtype=torch.bool)]
    
    embed_dim = embeddings.shape[1] # Embedding dimension
    num_items = dataset.num(item_id_field)
    output = torch.zeros(num_items, embed_dim, dtype=embeddings.dtype)
    # Convert all tokens into internal ids to grab the indices associated with each item
    internal_ids = torch.as_tensor(dataset.token2id(item_id_field, ext_ids), dtype=torch.long)
    # External ids and embeddings were saved to be consistent index-wise
    output[internal_ids] = embeddings
    return output

def load_price_bins(
    pickle_path: str | Path,
    dataset,
    item_id_field: str = "item_id",
    n_bins: int=32,
) -> torch.Tensor:
    """Create and return an index tensor that can be thrown into nn.Embedding

    Args:
        pickle_path (str | Path): Path to data/amazon_m2/price_bins.pt
        dataset (RecBole Dataset or SequentialDataset): Underlying dataset
        item_id_field (str, optional): RecBole field name holding the item IDs. Defaults to "item_id".
        n_bins (int, optional): Number of price bins. Defaults to 32.

    Returns:
        torch.Tensor: Shape (num_items, ) Row 0's value is n_bins to be used as a reserved sentinel, rows 1..num_items-1 are the bin indices of each respective item
    """
    # Load in ids and price bins and filter out missing items
    pickle_path = Path(pickle_path)
    payload = torch.load(pickle_path)
    ext_ids, bins = payload['item_ids'], payload['bin_idx']
    known_tokens = set(dataset.field2token_id[item_id_field])
    present = [1 if id in known_tokens else 0 for id in ext_ids]
    n_total = len(ext_ids)
    n_missing = n_total - sum(present)
    if n_missing > 0:
        print(f"Missing {n_missing} out of {n_total} items in the input dataset...", flush=True)
    ext_ids = [ext_ids[i] for i in range(n_total) if present[i]]
    bins = bins[torch.as_tensor(present, dtype=torch.bool)]
    
    assert bins.dtype == torch.long, f"Expected bins dtype of {torch.long} but received type {bins.dtype}"
    num_items = dataset.num(item_id_field)
    
    # Every row should start at the value 'n_bins', and will be overrided later
    output = torch.full((num_items,), n_bins, dtype=torch.long) # This way items who for some reason don't have a price bin associated with them will be thrown into bin 'n_bins' (though we designed this pipeline to not let that happen)
    # Bind each item to its internal ID
    internal_ids = torch.as_tensor(dataset.token2id(item_id_field, ext_ids), dtype=torch.long)
    # Each item will be assigned its corresponding price bin
    output[internal_ids] = bins
    return output