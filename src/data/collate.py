import torch
from torch_geometric.data import Batch


def collate_fn(batch):
    """
    Collate function for batching sequences of graphs.

    Converts:
        List[(seq_of_graphs, label, time)]
    into:
        List[Batch], labels tensor, time tensor
    """

    sequences, labels, times = zip(*batch)

    # Number of timesteps in sequence
    seq_len = len(sequences[0])

    # Batch graphs timestep-wise
    batched_sequences = []
    for t in range(seq_len):
        graphs_t = [seq[t] for seq in sequences]
        batched_sequences.append(Batch.from_data_list(graphs_t))

    labels = torch.tensor(labels, dtype=torch.long)
    times  = torch.tensor(times, dtype=torch.float32)

    return batched_sequences, labels, times
