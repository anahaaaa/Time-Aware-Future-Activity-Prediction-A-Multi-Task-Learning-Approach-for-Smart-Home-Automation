import numpy as np
import torch
import pandas as pd
from typing import Tuple

from configs.config import DataConfig


def build_sensor_graph(
    df: pd.DataFrame,
    cfg: DataConfig,
    num_sensors: int,
    top_k: int = 5,
    min_count: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a sensor graph based on transition probabilities.

    Nodes: Sensors
    Edges: Frequent transitions between sensors

    Args:
        df: Preprocessed dataframe
        cfg: Config object
        num_sensors: Number of unique sensors
        top_k: Top neighbors per sensor
        min_count: Minimum transition count to consider edge

    Returns:
        edge_index: [2, num_edges]
        edge_weight: [num_edges]
    """

   
    # Transition Count Matrix
    
    transition_counts = np.zeros(
        (num_sensors, num_sensors), dtype=np.float32
    )

    sensor_ids = df[cfg.SENSOR_COL + "_id"].values

    if len(sensor_ids) < 2:
        raise ValueError("Not enough data to build transitions")

    s1 = sensor_ids[:-1]
    s2 = sensor_ids[1:]

    np.add.at(transition_counts, (s1, s2), 1)

    
    # Normalize → Transition Probabilities
  
    row_sums = transition_counts.sum(axis=1, keepdims=True) + 1e-6
    transition_probs = transition_counts / row_sums

  
    # Build Edge List

    edges = set()
    weights = {}

    
    for i in range(num_sensors):
        edges.add((i, i))
        weights[(i, i)] = 1.0

    
    # Top-K Neighbor Selection

    for i in range(num_sensors):
        probs = transition_probs[i].copy()
        probs[i] = 0  

        # Get top-k indices
        top_k_idx = np.argsort(probs)[::-1][:top_k]

        for j in top_k_idx:
            if transition_counts[i, j] >= min_count:
                edges.add((i, j))
                edges.add((j, i))  # undirected graph

                weights[(i, j)] = float(transition_probs[i, j])
                weights[(j, i)] = float(transition_probs[j, i])


    # Convert to PyTorch format
    
    edges = list(edges)

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]

    edge_wt = [
        weights.get((r, c), 0.0) for r, c in zip(row, col)
    ]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(edge_wt, dtype=torch.float32)

   
    # Logging

    density = len(edges) / (num_sensors * num_sensors)

    print("\n🔗 Sensor Graph Summary")
    print("-" * 40)
    print(f"Nodes        : {num_sensors}")
    print(f"Edges        : {len(edges)}")
    print(f"Density      : {density:.4f}")
    print(f"Top-K        : {top_k}")
    print(f"Min Count    : {min_count}")

    return edge_index, edge_weight
