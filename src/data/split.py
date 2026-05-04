import pandas as pd
from typing import Tuple
from configs.config import DataConfig


def temporal_split(
    df: pd.DataFrame,
    cfg: DataConfig,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform time-aware split of dataset into train/val/test.

    Ensures chronological order is preserved (no data leakage).

    Args:
        df: Input dataframe
        cfg: Configuration object
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set

    Returns:
        train_df, val_df, test_df
    """

 
    # Validation
   
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if cfg.TIMESTAMP_COL not in df.columns:
        raise KeyError(f"{cfg.TIMESTAMP_COL} not found in dataframe")

   
    # Sort by time
 
    df = df.sort_values(cfg.TIMESTAMP_COL).reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError("Empty dataframe cannot be split")

    
    # Compute split indices
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

  
    # Split
  
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)

  
    # Logging
  
    print("\n Temporal Split Summary")
    print("-" * 40)
    print(f"Total samples : {n:,}")
    print(f"Train samples : {len(train_df):,}")
    print(f"Val samples   : {len(val_df):,}")
    print(f"Test samples  : {len(test_df):,}")

    print("\n Time Ranges")
    print(f"Train : {train_df[cfg.TIMESTAMP_COL].min()} → {train_df[cfg.TIMESTAMP_COL].max()}")
    print(f"Val   : {val_df[cfg.TIMESTAMP_COL].min()} → {val_df[cfg.TIMESTAMP_COL].max()}")
    print(f"Test  : {test_df[cfg.TIMESTAMP_COL].min()} → {test_df[cfg.TIMESTAMP_COL].max()}")

    return train_df, val_df, test_df
