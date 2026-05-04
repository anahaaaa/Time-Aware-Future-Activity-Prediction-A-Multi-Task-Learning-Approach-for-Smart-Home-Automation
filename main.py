# main.py

import argparse
import pandas as pd

from configs.config import DataConfig
from src.utils.seed import set_seed

# Data loading
from src.data.loader import load_csvs
from src.data.parser import process_dataset

from src.data.split import temporal_split

def load_data(args, cfg):
    """
    Load dataset based on input type.
    """
    if args.input_type == "csv":
        print(f"\n[1/3] Loading CSV data from: {args.path}")
        df = load_csvs(args.path)

    elif args.input_type == "txt":
        print(f"\n[1/3] Loading raw TXT dataset from: {args.path}")
        df = process_dataset(args.path)

    else:
        raise ValueError("input_type must be 'csv' or 'txt'")

    # Ensure timestamp format
    df[cfg.TIMESTAMP_COL] = pd.to_datetime(
        df[cfg.TIMESTAMP_COL], errors="coerce"
    )
    df = df.dropna(subset=[cfg.TIMESTAMP_COL])

    print(f"Loaded {len(df):,} rows")
    return df


def main():

    # Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path to dataset (csv glob or txt file)")
    parser.add_argument("--input_type", type=str, default="csv",
                        choices=["csv", "txt"])
    args = parser.parse_args()

    
    # Config + Seed
  
    cfg = DataConfig()
    set_seed(cfg.SEED)

 
    # Load Data
    
    df = load_data(args, cfg)
    print("\nData loading completed")

    # Preprocessing

    train_df, encoders = preprocess(train_df, cfg, fit=True)

    # Computing activity duration stats from train set
    activity_remaining_stats = compute_activity_remaining_stats(train_df, cfg)

    val_df, _  = preprocess(val_df,  cfg, encoders=encoders, fit=False)
    test_df, _ = preprocess(test_df, cfg, encoders=encoders, fit=False)
    # -----------------------------
    print("\n[TODO] Preprocessing...")
    print("[TODO] Graph construction...")
    print("[TODO] Sequence building...")
    print("[TODO] Model training...")
    print("[TODO] Evaluation...")


if __name__ == "__main__":
    main()
