import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
from typing import Optional

from configs.config import DataConfig


def window_to_graph(
    window: pd.DataFrame,
    next_event: pd.Series,
    cfg: DataConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_sensors: int,
    activity_remaining_stats: Optional[dict] = None
) -> Data:
    """
    Convert a time window into a graph representation.

    Node features include:
    - Sensor state & frequency
    - Temporal encodings
    - Activity progression signals
    - Statistical priors (optional)

    Args:
        window: Sliding window dataframe
        next_event: Next event (target)
        cfg: Configuration object
        edge_index: Graph connectivity
        edge_weight: Edge weights
        num_sensors: Number of sensors
        activity_remaining_stats: Optional duration priors

    Returns:
        PyTorch Geometric Data object
    """

  
    # Current activity & elapsed time
 
    curr_act = window[cfg.ACTIVITY_COL + "_id"].iloc[-1]

    start_idx = len(window) - 1
    for i in range(len(window) - 2, -1, -1):
        if window[cfg.ACTIVITY_COL + "_id"].iloc[i] != curr_act:
            break
        start_idx = i

    start_time = window[cfg.TIMESTAMP_COL].iloc[start_idx]
    end_time = window[cfg.TIMESTAMP_COL].iloc[-1]

    elapsed_min = (end_time - start_time).total_seconds() / 60.0
    elapsed = np.log1p(elapsed_min)

    elapsed_vec = np.full(num_sensors, elapsed, dtype=np.float32)

   
    # Activity progression features
  
    if activity_remaining_stats is not None:
        act_stat = activity_remaining_stats.get(int(curr_act), {})
        typical_duration = float(act_stat.get("median", 5.0))
    else:
        typical_duration = 5.0

    progress_ratio = np.clip(
        elapsed_min / (typical_duration + 1e-6), 0.0, 5.0
    )

    expected_remaining_min = max(0.0, typical_duration - elapsed_min)

    progress_vec = np.full(num_sensors, progress_ratio, dtype=np.float32)
    expected_remain_vec = np.full(
        num_sensors, np.log1p(expected_remaining_min), dtype=np.float32
    )
    typical_duration_vec = np.full(
        num_sensors, np.log1p(typical_duration), dtype=np.float32
    )

 
    # Window-level features
   
    window_duration = (
        window[cfg.TIMESTAMP_COL].iloc[-1] -
        window[cfg.TIMESTAMP_COL].iloc[0]
    ).total_seconds() / 60.0

    event_rate = np.log1p(len(window) / max(window_duration, 1e-5))
    event_rate_vec = np.full(num_sensors, event_rate, dtype=np.float32)

    # Time-of-day bucket
    hour = window[cfg.TIMESTAMP_COL].iloc[-1].hour
    bucket = 0 if hour < 8 else (1 if hour < 16 else 2)
    bucket_vec = np.full(num_sensors, bucket, dtype=np.float32)

    
    # Sensor state features

    sensor_ids = window[cfg.SENSOR_COL + "_id"].values
    sensor_vals = window[cfg.VALUE_COL].values

    sensor_values = np.zeros(num_sensors, dtype=np.float32)
    sensor_counts = np.zeros(num_sensors, dtype=np.float32)

    for sid, val in zip(sensor_ids, sensor_vals):
        sensor_values[sid] = val
        sensor_counts[sid] += 1

    sensor_counts /= max(len(window), 1)

   
    # Temporal encodings
   
    hour_sin_vec = np.full(num_sensors, window["hour_sin"].iloc[-1], dtype=np.float32)
    hour_cos_vec = np.full(num_sensors, window["hour_cos"].iloc[-1], dtype=np.float32)
    day_sin_vec  = np.full(num_sensors, window["day_sin"].iloc[-1], dtype=np.float32)
    day_cos_vec  = np.full(num_sensors, window["day_cos"].iloc[-1], dtype=np.float32)

    dt_vec = np.full(num_sensors, window["log_dt"].iloc[-1], dtype=np.float32)
    prev_dur_vec = np.full(num_sensors, window["log_prev_dur"].iloc[-1], dtype=np.float32)

    delta_minutes = (
        window[cfg.TIMESTAMP_COL].iloc[-1] -
        window[cfg.TIMESTAMP_COL].iloc[0]
    ).total_seconds() / 60.0

    delta_vec = np.full(num_sensors, np.log1p(delta_minutes), dtype=np.float32)

  
    # Combine node features
   
    node_features = np.stack([
        sensor_values,
        sensor_counts,
        hour_sin_vec,
        hour_cos_vec,
        day_sin_vec,
        day_cos_vec,
        delta_vec,
        dt_vec,
        prev_dur_vec,
        elapsed_vec,
        event_rate_vec,
        bucket_vec,
        progress_vec,
        expected_remain_vec,
        typical_duration_vec,
    ], axis=1)

    x = torch.from_numpy(node_features)

    
    # Target labels
    
    next_label = int(next_event[cfg.ACTIVITY_COL + "_id"])
    prev_act_id = window["prev_activity_id"].iloc[-1]

    
    # Graph object
    
    g = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1)
    )

    g.y = torch.tensor(next_label, dtype=torch.long)

    # Placeholder (overwritten later in sequence builder)
    g.y_time = torch.tensor(0.0, dtype=torch.float32)

    g.prev_activity = torch.tensor(prev_act_id, dtype=torch.long)

    return g
