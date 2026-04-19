from pathlib import Path
import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_inter", type=Path, default=Path("data/amazon_m2/amazon_m2.test.inter"))
    parser.add_argument("--locale_map", type=Path, default=Path("data/amazon_m2/locale_map.parquet"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/amazon_m2/"))
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Read test inter file and locale dataframe
    test_df = pd.read_csv(args.test_inter, sep='\t')
    locale_df = pd.read_parquet(args.locale_map)
    locale_df = locale_df.rename(columns={'session_id': 'session_id:token'})
    # Merge will have every original test column and a locale column
    merged_df = pd.merge(test_df, locale_df, how='left', on='session_id:token')
    # Fail on unmatched rows - every row should have a locale
    if merged_df['locale'].isna().any():
        raise ValueError(f"Found {merged_df['locale'].isna().sum()} sessions without locales...")
    # Find unique locales
    locales = merged_df['locale'].unique()
    for locale in locales:
        relevant_sessions = merged_df[merged_df['locale'] == locale]
        out_df = relevant_sessions.drop(columns='locale')
        output_path = args.output_dir / f"amazon_m2.test_{locale.lower()}.inter"
        out_df.to_csv(output_path, sep="\t", index=False)
        print(f"{locale}: {len(out_df)}", flush=True)
    print(f"TOTAL: {len(merged_df)}", flush=True)
        
    
    
if __name__=="__main__":
    main()