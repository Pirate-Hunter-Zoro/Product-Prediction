from pathlib import Path
import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/amazon_m2/locale_map.parquet"))
    args = parser.parse_args()
    sessions_df = pd.read_csv(Path("data/sessions_train.csv"), nrows=args.nrows)
    sessions_df['id'] = sessions_df.index
    # Grab the two columns we care about
    lookup_df = sessions_df[['id', 'locale']].rename(columns={'id': 'session_id'})
    # Ensure output directory exists
    os.makedirs(args.output.parent, exist_ok=True)
    lookup_df.to_parquet(args.output, index=False)
    print(f"Finished outputting locale_map to {args.output} with {len(lookup_df)} rows", flush=True)
    
if __name__=="__main__":
    main()