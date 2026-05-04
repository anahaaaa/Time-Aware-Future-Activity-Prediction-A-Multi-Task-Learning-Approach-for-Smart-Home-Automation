from torch.utils.data import Dataset
from typing import List, Tuple


class SequenceDataset(Dataset):
    """
    Dataset for sequence-based graph data.

    Each item:
        (sequence_of_graphs, activity_label, time_target)
    """

    def __init__(self, sequences: List[Tuple]):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
