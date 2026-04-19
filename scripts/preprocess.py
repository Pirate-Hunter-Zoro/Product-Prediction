import os
import re
from pathlib import Path
import pandas as pd
import argparse

SHUFFLE_SEED = 42 # The meaning of life

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', type=int, default=None)
    args = parser.parse_args()
    
    sessions_df = pd.read_csv(Path("data/sessions_train.csv"), nrows=args.nrows)
    sessions_df['id'] = sessions_df.index
    
     # .inter file
    n = len(sessions_df)
    train_end = int(n*0.8)
    valid_end = int(n*0.9)
    # Shuffle order
    sessions_df = sessions_df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
    
    items = sessions_df['prev_items'].apply(lambda s: re.findall(r"[A-Z0-9]{10}", s))
    items = items.apply(lambda items_list: " ".join(items_list))
    
    for split in [('train', (0, train_end)), ('valid', (train_end, valid_end)), ('test', (valid_end, n))]:
        split_name = split[0]
        (start, end) = split[1]
        split_df = sessions_df.iloc[start:end]
        split_items = items.iloc[start:end]
        inter_path = Path(f"data/amazon_m2/amazon_m2.{split_name}.inter")
        os.makedirs(inter_path.parent, exist_ok=True)
        output_df = pd.DataFrame({                                                                                                            
            "session_id:token": split_df['id'],
            "item_id_list:token_seq": split_items,                                                                           
            "item_id:token": split_df['next_item']                                                                                           
        }) 
        output_df.to_csv(inter_path, sep="\t", index=False)
        print("Created .inter files...", flush=True)

if __name__=="__main__":
    main()