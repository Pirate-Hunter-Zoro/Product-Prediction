import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import argparse
import os

def load_text_column(parquet_path: Path, column: str) -> tuple[list[str], list[str]]:
    """Load column values of parquet path

    Args:
        parquet_path (Path): Information on all items
        column (str): Column associated with the values we want to embed

    Returns:
        tuple[list[str], list[str]]: item_ids, column_vals
    """
    df = pd.read_parquet(parquet_path)
    item_ids, column_vals = df['item_id'].tolist(), df[column].tolist()
    assert len(item_ids) == len(column_vals), f"Mismatching item ID list and {column} list of lengths {len(item_ids)} and {len(column_vals)} respectively"
    return (item_ids, column_vals)

def encode_texts(column_vals: list[str], model_path: Path, device: str, batch_size: int) -> torch.FloatTensor:
    """Return encodings of all the given string column values

    Args:
        column_vals (list[str]): Full list of column value strings
        model_path (Path): Path to locally downloaded embedding model
        device (str): cuda or cpu
        batch_size (int): Size of each batch to embed

    Returns:
        torch.FloatTensor: (len(titles), 384), dtype=torch.float32 on CPU
    """
    model = SentenceTransformer(str(model_path), device=device)
    encodings = model.encode(column_vals, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    assert encodings.shape == (len(column_vals), 384)
    return torch.from_numpy(encodings).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path("data/amazon_m2/item_attributes.parquet"))
    parser.add_argument('--model', type=Path, default=Path("/media/studies/ehr_study/analysis/mferguson/models/paraphrase-multilingual-MiniLM-L12-v2/"))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--column', type=str, required=True, choices={"title", "brand", "color"})
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ids, column_vals = load_text_column(args.input, args.column)
    if args.nrows is not None:
        column_vals = column_vals[:args.nrows]
        ids = ids[:args.nrows]
    if args.output is None:
        output = Path(f"data/amazon_m2/{args.column}_embeddings.pt")
    else:
        output = args.output
    # If the user tries to set the device to cuda and it fails, they should know that - we're not just gonna default to CPU
    embeddings = encode_texts(column_vals, args.model, device=args.device, batch_size=args.batch_size)
    results = {
        "item_ids": ids,
        "embeddings": embeddings
    }
    os.makedirs(output.parent, exist_ok=True)
    torch.save(results, output)
    print(f"Wrote {output} shape={tuple(embeddings.shape)}")
    
    if args.debug:
        d = torch.load(output, weights_only=False)                                                              
        print(sorted(d.keys()))                                                                                                               
        print(len(d['item_ids']), d['embeddings'].shape, d['embeddings'].dtype)                                                               
        print('norms:', d['embeddings'].norm(dim=1)[:3])

if __name__=="__main__":
    main()