"""Tools for generating and handling segments."""

import numpy as np


def binarise(
    input: np.ndarray,
    on_threshold: float = 0.5,
    off_threshold: float | None = None,
    min_duration_off: float | None = None,
    min_duration_on: float | None = None,
) -> np.ndarray:
    """Binarise an input array based on thresholds and minimum durations.

    Args:
        input: 1D numpy array of float values to be binarised.
        on_threshold: Values above this threshold are considered "on".
        off_threshold: Values below this threshold are considered "off".
        min_duration_off: Minimum duration (in samples) for "off" segments.
        min_duration_on: Minimum duration (in samples) for "on" segments.

    Returns:
        output (np.ndarray): A boolean numpy array indicating "on" (1) and
            "off" (0) states.
    """
    if input.ndim != 1:
        msg = "Input array must be 1D"
        raise ValueError(msg)

    if off_threshold is None:
        binary_array = input >= on_threshold
    else:
        if on_threshold < off_threshold:
            msg = "On threshold must be greater than or equal to off threshold"
            raise ValueError(msg)
        binary_array = np.zeros_like(input, dtype=bool)
        current = input[0] >= 0.5 * (on_threshold + off_threshold)
        for n, val in enumerate(input):
            if val > on_threshold:
                binary_array[n] = True
            elif val < off_threshold:
                binary_array[n] = False
            else:
                binary_array[n] = current
            current = binary_array[n]

    if min_duration_off is not None or min_duration_on is not None:
        msg = "Minimum duration handling not yet implemented"
        raise NotImplementedError(msg)

    return binary_array
