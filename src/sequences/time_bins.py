import numpy as np
from typing import List, Tuple


def compute_quantile_bins(
    sequences: List[Tuple],
    n_bins: int = 10,
    max_time: float = 120.0
) -> np.ndarray:
    """
    Compute quantile-based time bins from training sequences.

    Converts log-time targets back to minutes and builds bins
    such that each bin has roughly equal number of samples.

    Args:
        sequences: List of (seq, label, log_time)
        n_bins: Number of bins
        max_time: Upper cap for time (minutes)

    Returns:
        edges: Array of bin edges (length = n_bins + 1)
    """

    if len(sequences) == 0:
        raise ValueError("Sequences list is empty")

    # Convert log time → minutes
    times = np.array([np.expm1(t) for _, _, t in sequences])

    # Compute quantiles
    edges = np.percentile(
        times,
        np.linspace(0, 100, n_bins + 1)
    )

    # Ensure stable boundaries
    edges[0] = 0.0
    edges[-1] = max_time + 1.0  # slight buffer

    print("\n⏱️ Time Bin Edges (minutes)")
    print("-" * 40)
    print(np.round(edges, 2))

    return edges


def assign_time_bins(
    sequences: List[Tuple],
    bin_edges: np.ndarray
) -> List[Tuple]:
    """
    Assign each sequence to a time bin.

    Args:
        sequences: List of (seq, label, log_time)
        bin_edges: Output from compute_quantile_bins()

    Returns:
        List of (seq, label, bin_id)
    """

    if len(bin_edges) < 2:
        raise ValueError("Invalid bin_edges")

    binned_sequences = []

    for seq, label, t_log in sequences:
        t = float(np.expm1(t_log))  # back to minutes

        # Find bin index
        bin_id = int(np.searchsorted(bin_edges[1:], t))

        # Safety clamp
        bin_id = min(bin_id, len(bin_edges) - 2)

        binned_sequences.append((seq, label, bin_id))

    return binned_sequences
