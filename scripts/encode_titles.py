import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import argparse
import os

def load_titles(parquet_path: Path) -> tuple[list[str], list[str]]:
    """Load titles of parquet path

    Args:
        parquet_path (Path): Information on all items

    Returns:
        tuple[list[str], list[str]]: item_ids, titles
    """
    df = pd.read_parquet(parquet_path)
    item_ids, titles = df['item_id'].tolist(), df['title'].tolist()
    assert len(item_ids) == len(titles), f"Mismatching item ID list and title list of lengths {len(item_ids)} and {len(titles)} respectively"
    return (item_ids, titles)

def encode_titles(titles: list[str], model_path: Path, device: str, batch_size: int) -> torch.FloatTensor:
    """Return encodings of all the given string titles

    Args:
        titles (list[str]): Full list of title strings
        model_path (Path): Path to locally downloaded embedding model
        device (str): cuda or cpu
        batch_size (int): Size of each batch to embed

    Returns:
        torch.FloatTensor: (len(titles), 384), dtype=torch.float32 on CPU
    """
    model = SentenceTransformer(str(model_path), device=device)
    encodings = model.encode(titles, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    assert encodings.shape == (len(titles), 384)
    return torch.from_numpy(encodings).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path("data/amazon_m2/item_attributes.parquet"))
    parser.add_argument('--output', type=Path, default=Path("data/amazon_m2/title_embeddings.pt"))
    parser.add_argument('--model', type=Path, default=Path("/media/studies/ehr_study/analysis/mferguson/models/paraphrase-multilingual-MiniLM-L12-v2/"))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()
    ids, titles = load_titles(args.input)
    if args.nrows is not None:
        titles = titles[:args.nrows]
        ids = ids[:args.nrows]
    # If the user tries to set the device to cuda and it fails, they should know that - we're not just gonna default to CPU
    embeddings = encode_titles(titles, args.model, device=args.device, batch_size=args.batch_size)
    results = {
        "item_ids": ids,
        "embeddings": embeddings
    }
    os.makedirs(args.output.parent, exist_ok=True)
    torch.save(results, args.output)
    print(f"Wrote {args.output} shape={tuple(embeddings.shape)}")
    
if __name__=="__main__":
    main()
    
    d = torch.load('data/amazon_m2/title_embeddings.pt', weights_only=False)                                                              
    print(sorted(d.keys()))                                                                                                               
    print(len(d['item_ids']), d['embeddings'].shape, d['embeddings'].dtype)                                                               
    print('norms:', d['embeddings'].norm(dim=1)[:3]) 