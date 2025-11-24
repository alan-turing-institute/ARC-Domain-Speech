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

    if min_duration_off is not None:
        start_idx = 0
        status = binary_array[0]
        for n in range(1, len(binary_array)):
            if binary_array[n] != status:
                if not status and n - start_idx < min_duration_off:
                    binary_array[start_idx:n] = True
                start_idx = n
                status = binary_array[n]

        if not status and len(binary_array) - start_idx < min_duration_off:
            binary_array[start_idx:] = True

    if min_duration_on is not None:
        start_idx = 0
        status = binary_array[0]
        for n in range(1, len(binary_array)):
            if binary_array[n] != status:
                if status and n - start_idx < min_duration_on:
                    binary_array[start_idx:n] = False
                start_idx = n
                status = binary_array[n]

        if status and len(binary_array) - start_idx < min_duration_on:
            binary_array[start_idx:] = False

    return binary_array


def segment_times(
    binary_array: np.ndarray, time_start: float, time_step: float
) -> list[tuple[float, float]]:
    """Convert a binary array to a list of time segments.

    Args:
        binary_array: 1D numpy array of boolean values indicating "on" (True)
            and "off" (False) states.
        time_start: (float): Value for the first timestamp.
        time_step: (float): Time difference between consecutive timestamps.

    Returns:
        segments (list of tuples): List of (start_time, end_time) tuples for
            each "on" segment.
    """
    timestamps = time_start - time_step / 2 + time_step * np.arange(len(binary_array))

    segments = []
    start_time = 0.0
    in_segment = binary_array[0]

    for n, val in enumerate(binary_array):
        if val and not in_segment:
            in_segment = True
            start_time = timestamps[n]
        elif not val and in_segment:
            in_segment = False
            end_time = timestamps[n]
            segments.append((start_time, end_time))

    if in_segment:
        end_time = (len(binary_array) - 1) * time_step + time_start * 2
        segments.append((start_time, end_time))

    return segments
