import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional

from configs.config import DataConfig


def build_sequences_global(
    df: pd.DataFrame,
    cfg: DataConfig,
    num_sensors: int,
    seq_len: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    other_id: Optional[int] = None,
    activity_remaining_stats: Optional[dict] = None
) -> List[Tuple[List, int, float]]:
    """
    Build sequences of graph snapshots for activity prediction.

    Each sequence:
        - Input: sequence of graph objects
        - Target: next activity + time-to-next-activity

    Args:
        df: Preprocessed dataframe
        cfg: Configuration object
        num_sensors: Number of sensors
        seq_len: Length of sequence
        edge_index: Graph edges
        edge_weight: Edge weights
        other_id: ID for "other" class (optional filtering)
        activity_remaining_stats: Optional duration priors

    Returns:
        List of (sequence, next_activity, time_delta)
    """

    if edge_index is None:
        raise ValueError("edge_index must be provided")

    edge_index = edge_index.long().contiguous()
    edge_weight = edge_weight.float().contiguous()

    assert edge_index.max() < num_sensors
    assert edge_index.min() >= 0

    df = df.reset_index(drop=True)
    n = len(df)

    all_sequences = []
    skipped_no_change = 0
    skipped_other = 0

    activity_vals = df[cfg.ACTIVITY_COL].values
    timestamps = df[cfg.TIMESTAMP_COL].values


    # Detect activity change points

    change_indices = np.where(
        np.concatenate([[True], activity_vals[1:] != activity_vals[:-1]])
    )[0]

    graphs = []


    # Sliding window → Graph construction
  
    for start in range(0, n - cfg.WINDOW_SIZE - 1, cfg.STRIDE):

        window_end_idx = start + cfg.WINDOW_SIZE - 1
        next_idx = start + cfg.WINDOW_SIZE

        window = df.iloc[start:next_idx]
        next_event = df.iloc[next_idx]

        curr_act = activity_vals[window_end_idx]
        curr_time = timestamps[window_end_idx]

      
        # Find next activity change
      
        pos = np.searchsorted(change_indices, next_idx, side="left")

        found = False
        time_to_next_start_min = 0.0

        for p in range(pos, len(change_indices)):
            candidate_idx = change_indices[p]

            if candidate_idx >= n:
                break

            if activity_vals[candidate_idx] != curr_act:
                t_next = timestamps[candidate_idx]

                delta_sec = float(
                    (t_next - curr_time) / np.timedelta64(1, "s")
                )

                time_to_next_start_min = float(
                    np.clip(delta_sec / 60.0, 0.0, 120.0)
                )

                found = True
                break

        if not found:
            skipped_no_change += 1
            continue

        
        # Convert window → graph
     
        g = window_to_graph(
            window,
            next_event,
            cfg,
            edge_index,
            edge_weight,
            num_sensors,
            activity_remaining_stats=activity_remaining_stats
        )

        # Target-> time-to-next-activity
        g.y_time = torch.tensor(
            np.log1p(time_to_next_start_min),
            dtype=torch.float32
        )

        g.time = curr_time
        graphs.append(g)

  \
    # Sequence creation
   
    total_graphs = len(graphs)

    if total_graphs < seq_len + 1:
        print(f"⚠️ Not enough graphs ({total_graphs}) for seq_len={seq_len}")
        return []

    for i in range(total_graphs - seq_len):
        seq = graphs[i:i + seq_len]
        next_graph = graphs[i + seq_len]

        next_activity = next_graph.y.item()
        time_delta = next_graph.y_time.item()

        # Filtering
        if other_id is not None and next_activity == other_id:
            skipped_other += 1
            continue

        all_sequences.append((seq, next_activity, time_delta))

    
    # Logging
  
    print("\n Sequence Generation Summary")
    print("-" * 40)
    print(f"Input rows       : {n}")
    print(f"Graphs created   : {total_graphs}")
    print(f"Sequences        : {len(all_sequences)}")

    print(f"Skipped (no change) : {skipped_no_change}")
    print(f"Skipped (other)     : {skipped_other}")

    if all_sequences:
        times_min = np.array([np.expm1(t) for _, _, t in all_sequences])

        print("\n⏱️ Time-to-Next-Activity Stats (minutes)")
        print(f"Median : {np.median(times_min):.1f}")
        print(f"Mean   : {np.mean(times_min):.1f}")
        print(f"P10    : {np.percentile(times_min, 10):.1f}")
        print(f"P90    : {np.percentile(times_min, 90):.1f}")
        print(f"Max    : {np.max(times_min):.1f}")

    return all_sequences
