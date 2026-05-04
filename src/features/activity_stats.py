import numpy as np
import pandas as pd
from typing import Dict
from configs.config import DataConfig


def compute_activity_remaining_stats(
    df: pd.DataFrame,
    cfg: DataConfig
) -> Dict[int, Dict[str, float]]:
    """
    Compute statistical summaries of activity durations.

    For each activity class, calculates distribution statistics
    of how long that activity typically lasts.

    Args:
        df: Preprocessed dataframe (must contain activity_change)
        cfg: Configuration object

    Returns:
        stats: Dictionary mapping activity_id → statistics
    """

    stats = {}

  
    # Group by activity and activity segments
    
    groups = df.groupby([cfg.ACTIVITY_COL + "_id", "activity_change"])

    all_durations = {}

    for (act_id, run_id), group in groups:
        duration = (
            group[cfg.TIMESTAMP_COL].max() -
            group[cfg.TIMESTAMP_COL].min()
        ).total_seconds() / 60.0  # minutes

        if act_id not in all_durations:
            all_durations[act_id] = []

        all_durations[act_id].append(duration)

   
    # Compute statistics per activity

    for act_id, durations in all_durations.items():
        d = np.array(durations)

        stats[act_id] = {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
            "p90": float(np.percentile(d, 90)),
            "count": len(d)
        }

    # Logging 
 
    print("\n Activity Duration Statistics")
    print("-" * 40)
    print(f"Computed for {len(stats)} activity classes")

    return stats
