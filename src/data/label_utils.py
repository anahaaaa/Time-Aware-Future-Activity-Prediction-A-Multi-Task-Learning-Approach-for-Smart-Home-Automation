from typing import List, Tuple, Dict


def remap_classes(
    sequences: List[Tuple],
    class_map: Dict[int, int]
) -> List[Tuple]:
    """
    Remap activity labels to a new class index mapping.

    Filters out samples whose labels are not in class_map.

    Args:
        sequences: List of (sequence, label, time)
        class_map: Mapping from old_label → new_label

    Returns:
        Remapped sequence list
    """

    remapped = [
        (seq, class_map[label], time)
        for seq, label, time in sequences
        if label in class_map
    ]

    print(f"Remapped sequences: {len(remapped)}")

    return remapped
