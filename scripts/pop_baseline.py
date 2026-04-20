import argparse
import json
import math
from pathlib import Path
from collections import Counter
import pandas as pd

def build_popularity_ranking(train_path: Path, topk: int=100) -> tuple[list[str], Counter]:
    """Return most popular k items in decreasing order of popularity

    Args:
        train_path (Path): path to amazon_m2.train.inter
        topk (int): how many top items to retain

    Returns:
        tuple[list[str], Counter]: list of item_id strings, length==topk, most-popular first, as well as the count of all items
    """
    training_df = pd.read_csv(train_path, sep="\t")
    counter = Counter()
    for token_seq in training_df['item_id_list:token_seq']:
        # Initially the token sequence is a space separated string of item IDs
        counter.update(token_seq.split())
    # Count each session's target item
    counter.update(list(training_df['item_id:token']))
    top_items_with_counts = counter.most_common(topk)
    return ([t[0] for t in top_items_with_counts], counter)

def score_test_set(test_path: Path, ranking: list[str]) -> dict[str, float]:
    """Obtain next product prediction scores given the most popular items

    Args:
        test_path (Path): path to amazon_m2.test.inter
        ranking (list[str]): output of build_popularity_ranking

    Returns:
        dict[str, float]: {"mrr@100": float, "recall@100": float, "ndcg@100": float}
    """
    test_df = pd.read_csv(test_path, sep="\t")
    lookup_rank = {}
    for i, target_item in enumerate(ranking):
        lookup_rank[target_item] = i+1
    # Set up accumulators
    mrr_score = 0.0
    recall_score = 0.0
    ndcg_score = 0.0
    n = 0
    for target_item in test_df['item_id:token']:
        n += 1
        if target_item in lookup_rank:
            # One of the top 100 items
            rank = lookup_rank[target_item]
            mrr_score += 1.0 / rank # Contribution matters according to rank
            recall_score += 1.0 # Contribution only dependent on if in top 100 or not
            ndcg_score += 1.0 / math.log2(rank + 1)
    mrr_score /= n
    recall_score /= n
    ndcg_score /= n
    return {
        "mrr@100": mrr_score,
        "recall@100": recall_score,
        "ndcg@100": ndcg_score,
    }
          
def score_test_set_session_aware(test_path: Path, global_counter: Counter, topk: int) -> dict[str, float]:
    """Obtain next product prediction scores given the most popular items

    Args:
        test_path (Path): path to amazon_m2.test.inter
        global_counter (Counter): keeps track of the count of each product
        topk (int): top items to consider for popularity

    Returns:
        dict[str, float]: {"mrr@100": float, "recall@100": float, "ndcg@100": float}
    """
    df = pd.read_csv(test_path, sep="\t")
    mrr_score = 0.0
    recall_score = 0.0
    ndcg_score = 0.0
    global_fallback = [item[0] for item in global_counter.most_common(topk+500)]
    n = 0
    for history, target in zip(df['item_id_list:token_seq'], df['item_id:token']):
        n += 1
        tokens = history.split()
        session_counter = Counter(tokens)
        # Pad from global fallback if we do not have enough items
        token_set = set(tokens)
        fallback_idx = 0
        while len(token_set) < topk:
            token_set.add(global_fallback[fallback_idx])
            fallback_idx += 1
        tokens = list(token_set)
        # Sort by descending frequency in session count, tie breaking with global count
        tokens.sort(key=lambda x: (session_counter.get(x, 0), global_counter.get(x, 0)), reverse=True)
        tokens = tokens[:topk]
        ranks = {item: i+1 for i, item in enumerate(tokens)}
        if target in ranks:
            # One of the top 100 items
            rank = ranks[target]
            mrr_score += 1.0 / rank # Contribution matters according to rank
            recall_score += 1.0 # Contribution only dependent on if in top 100 or not
            ndcg_score += 1.0 / math.log2(rank + 1)
    mrr_score /= n
    recall_score /= n
    ndcg_score /= n
    return {
        "mrr@100": mrr_score,
        "recall@100": recall_score,
        "ndcg@100": ndcg_score,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/amazon_m2/amazon_m2.train.inter")
    parser.add_argument("--test", type=str, default="data/amazon_m2/amazon_m2.test.inter")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    
    ranking, counter = build_popularity_ranking(args.train, args.topk)
    metrics = score_test_set(args.test, ranking)
    print("Global Metrics:")
    print(json.dumps(metrics, indent=4), flush=True)
    metrics = score_test_set_session_aware(args.test, counter, args.topk)
    print("Session Metrics:")
    print(json.dumps(metrics, indent=4), flush=True)

if __name__=="__main__":
    main()