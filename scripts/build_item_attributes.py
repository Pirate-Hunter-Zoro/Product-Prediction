import argparse
from pathlib import Path
import pandas as pd
import os

KEEP_COLS = [
    "id",
    "title",
    "brand",
    "price",
    "color"
]

def build_item_attributes(
    products_csv: Path,
    inter_dir: Path,
    output: Path,
    nrows: int | None = None, 
):
    """Write .parquet describing product metadata

    Args:
        products_csv (Path): Raw product metadata file
        inter_dir (Path): Directory containing tran,valid,test .inter files
        output (Path): Where .parquet file is written
        nrows (int | None, optional): If set read this many rows, otherwise all rows. Defaults to None.
    """
    products = pd.read_csv(products_csv, nrows=nrows)
    products = products[KEEP_COLS] # Only the columns we care about
    products = products.rename(columns={"id": "item_id"})
    products[["title", "brand", "color"]] = products[["title", "brand", "color"]].fillna("")
    train_df, valid_df, test_df = pd.read_csv(inter_dir / "amazon_m2.train.inter", sep="\t"), \
                                    pd.read_csv(inter_dir / "amazon_m2.valid.inter", sep="\t"), \
                                        pd.read_csv(inter_dir / "amazon_m2.test.inter", sep="\t")
    vocab = set(train_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(train_df['item_id:token'].unique().tolist()) | \
        set(valid_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(valid_df['item_id:token'].unique().tolist()) | \
    set(test_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(test_df['item_id:token'].unique().tolist())
    
    # Discard nan values and empty string values
    vocab = {v for v in vocab if isinstance(v, str)}
    assert products["item_id"].dtype == object, f"Expected dtype {object} but found {products['item_id'].dtype}"
    
    # Overlap statistic
    vocab_size = len(vocab)
    # Drop indices and items of filtered_products refering to dropped rows when missing from vocab
    filtered_products = products[products['item_id'].isin(vocab)].reset_index(drop=True)
    filtered_products = filtered_products.drop_duplicates(subset="item_id", keep="first").reset_index(drop=True)
    
    print(f"Products in vocab: {len(filtered_products)}\n\n", flush=True)
    print(f"Vocab size: {vocab_size}\n\n", flush=True)
    print(f"Number of products: {len(products)}", flush=True)
    # Any vocab members not in the catalog are a problem - sessions reference it, we have to embed it somehow, and so we need to make a fallback
    print(f"Number of filtered products: {len(filtered_products)}", flush=True)
    
    print(f"Size of vocab missing from catalog: {vocab_size - filtered_products['item_id'].nunique()}", flush=True)
    print(f"Number of products that did not appear in vocab: {len(products)-len(filtered_products)}", flush=True)
    
    os.makedirs(output.parent, exist_ok=True)
    filtered_products.to_parquet(output, index=False)
    print(f"Wrote {len(filtered_products)} rows to {output}", flush=True)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Build per-item attribute parquet from products_train.csv filtered to RecBole vocab")
    parser.add_argument("--products-csv", type=Path, default=Path("data/products_train.csv"))
    parser.add_argument("--inter-dir", type=Path, default=Path("data/amazon_m2"))
    parser.add_argument("--output", type=Path, default=Path("data/amazon_m2/item_attributes.parquet"))
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    build_item_attributes(
        args.products_csv,
        args.inter_dir,
        args.output,
        args.nrows
    )