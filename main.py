# main.py

import argparse
import pandas as pd

from configs.config import DataConfig
from src.utils.seed import set_seed

# Data loading
from src.data.loader import load_csvs
from src.data.parser import process_dataset
from src.data.split import temporal_split

from src.features.activity_stats import compute_activity_remaining_stats
from src.graph.build_graph import build_sensor_graph

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

    # temporal split
    df = df.sort_values(cfg.TIMESTAMP_COL).reset_index(drop=True)
    train_df, val_df, test_df = temporal_split(df, cfg)

    # Preprocessing

    train_df, encoders = preprocess(train_df, cfg, fit=True)

    # Computing activity duration stats from train set
    activity_remaining_stats = compute_activity_remaining_stats(train_df, cfg)

    val_df, _  = preprocess(val_df,  cfg, encoders=encoders, fit=False)
    test_df, _ = preprocess(test_df, cfg, encoders=encoders, fit=False)

    num_sensors = len(encoders[cfg.SENSOR_COL].classes_)
    class_names = encoders[cfg.ACTIVITY_COL].classes_

    # Detect "other" class
    other_id = None
    for candidate in ["other", "o"]:
        if candidate in class_names:
            other_id = int(encoders[cfg.ACTIVITY_COL].transform([candidate])[0])
            print(f"Other class ID: {other_id} ({candidate})")
            break

    # Graph Construction

    train_edge_index, train_edge_weight = build_sensor_graph(
        train_df, cfg, num_sensors
    )


    # Build Sequences
  
    print("\n Building graph sequences…")
    
    train_sequences = build_sequences_global(
        train_df, cfg, num_sensors, seq_len,
        edge_index=train_edge_index,
        edge_weight=train_edge_weight,
        other_id=other_id,
        activity_remaining_stats=activity_remaining_stats
    )

    val_sequences = build_sequences_global(
        val_df, cfg, num_sensors, seq_len,
        edge_index=train_edge_index,
        edge_weight=train_edge_weight,
        other_id=other_id,
        activity_remaining_stats=activity_remaining_stats
    )
    
    test_sequences = build_sequences_global(
        test_df, cfg, num_sensors, seq_len,
        edge_index=train_edge_index,
        edge_weight=train_edge_weight,
        other_id=other_id,
        activity_remaining_stats=activity_remaining_stats
    )

    # Filter sequences
    train_sequences = [s for s in train_sequences if len(s[0]) == seq_len]
    val_sequences   = [s for s in val_sequences   if len(s[0]) == seq_len]
    test_sequences  = [s for s in test_sequences  if len(s[0]) == seq_len]

    random.shuffle(train_sequences)

    print(f"Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")


    # -----------------------------
    print("[TODO] Graph construction...")
    print("[TODO] Sequence building...")
    print("[TODO] Model training...")
    print("[TODO] Evaluation...")


if __name__ == "__main__":
    main()
