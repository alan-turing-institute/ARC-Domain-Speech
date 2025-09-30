"""Tools for handling annotations."""

__all__ = ("speaking_map",)

import numpy as np


def speaking_map(
    timestamps: list[float] | np.ndarray, annotations: list[tuple[float, float]]
) -> np.ndarray:
    """Generate speaking map from timestamps and frame centers."""
    if isinstance(timestamps, list):
        try:
            timestamps = np.array(timestamps).astype(np.float32)
        except Exception as err:
            msg = "Could not convert timestamps to numpy array"
            raise ValueError(msg) from err
    if timestamps.ndim != 1:
        msg = "Timestamps must be a 1D array"
        raise ValueError(msg)
    if not timestamps.size > 0:
        msg = "Timestamps array is empty"
        raise ValueError(msg)

    speaking_map = np.zeros((timestamps.size,), dtype=np.int16)

    for anno in annotations:
        if len(anno) != 2:
            msg = "Each annotation must be a tuple of (start, end)"  # type: ignore[unreachable]
            raise ValueError(msg)
        if anno[0] >= anno[1]:
            msg = "Annotation start time must be less than end time"
            raise ValueError(msg)

        speaking_map[(timestamps >= anno[0]) & (timestamps <= anno[1])] = 1

    return speaking_map
