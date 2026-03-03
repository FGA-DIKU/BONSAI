from typing import List
import torch


def normalize_segments(segments: List[int]) -> List[int]:
    """Normalize a list of segment IDs to be zero-based and contiguous.

    Takes a list of segment IDs that may have gaps or start from an arbitrary number,
    and normalizes them to start from 0 and increment by 1 while preserving their relative order.

    Example:
        >>> normalize_segments([5, 5, 8, 10, 8])
        [0, 0, 1, 2, 1]
    """
    segment_set = sorted(set(segments))
    correct_segments = list(range(len(segment_set)))
    converter = {k.item(): v for (k, v) in zip(segment_set, correct_segments)}

    return torch.tensor(
        [converter[segment.item()] for segment in segments], dtype=segments.dtype
    )
