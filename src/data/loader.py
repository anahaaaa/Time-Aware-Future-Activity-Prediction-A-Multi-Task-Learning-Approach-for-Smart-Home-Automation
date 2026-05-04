import pandas as pd
import glob
import os
from tqdm import tqdm


def load_csvs(csv_pattern: str) -> pd.DataFrame:
    """
    Load and concatenate CSV files from a glob pattern.
    """
    files = glob.glob(csv_pattern)

    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {csv_pattern}")

    dfs = []
    for f in tqdm(files, desc="Loading CSVs"):
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(files)} files | {len(df):,} rows")
    return df
