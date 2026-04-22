import argparse
from pathlib import Path
import pandas as pd

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
    products["title"].fillna("", inplace=True)
    products["brand"].fillna("", inplace=True)
    products["color"].fillna("", inplace=True)
    train_df, valid_df, test_df = pd.read_csv(inter_dir / "amazon_m2.train.inter", sep="\t"), \
                                    pd.read_csv(inter_dir / "amazon_m2.valid.inter", sep="\t"), \
                                        pd.read_csv(inter_dir / "amazon_m2.test.inter", sep="\t")
    vocab = set(train_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(train_df['item_id:token'].unique().tolist()) | \
        set(valid_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(valid_df['item_id:token'].unique().tolist()) | \
    set(test_df['item_id_list:token_seq'].str.split(" ").explode().unique().tolist()) | set(test_df['item_id:token'].unique().tolist())
    