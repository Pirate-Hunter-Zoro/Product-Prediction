import argparse
import json
import math
from pathlib import Path
from collections import Counter
import pandas as pd

def build_popularity_ranking(train_path: Path, topk: int=100) -> list[str]:
    """Return most popular k items in decreasing order of popularity

    Args:
        train_path (Path): path to amazon_m2.train.inter
        topk (int): how many top items to retain

    Returns:
        list[str]: list of item_id strings, length==topk, most-popular first
    """
    training_df = pd.read_csv(train_path, sep="\t")
    counter = Counter()
    for token_seq in training_df['item_id_list:token_seq']:
        # Initially the token sequence is a space separated string of item IDs
        counter.update(token_seq.split())
    # Count each session's target item
    counter.update(list(training_df['item_id:token']))
    top_items_with_counts = counter.most_common(topk)
    return [t[0] for t in top_items_with_counts]

def score_test_set(test_path: Path, ranking: list[str]) -> dict[str, float]:
    """Obtain next product prediction scores given the most popular items

    Args:
        test_path (Path): path to amazon_m2.test.inter
        ranking (list[str]): output of build_popularity_ranking

    Returns:
        dict[str, float]: {"mrr@100": float, "recall@100": float, "ndcg@100": float}
    """
    test_df = pd.read_csv(test_path, sep="\t")
    

def main():
    pass

if __name__=="__main__":
    main()