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
        on_threshold: Values greater than or equal to this threshold are "on".
        off_threshold: Values less than or equal to this threshold are "off".
        min_duration_off: Minimum duration (in seconds) for "off" segments.
        min_duration_on: Minimum duration (in seconds) for "on" segments.

    Returns:
        binary_array (np.ndarray): A boolean numpy array indicating speech (True)
            and non-speech (False) frames.
    """
    if input.size == 0:
        msg = "Input array is empty"
        raise ValueError(msg)

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
        binary_array (np.ndarray): 1D numpy array of boolean values indicating
            speech (True) and non-speech (False) frames.
        time_start (float): Center time for the first frame.
        time_step (float): Time difference between consecutive frames.

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


def segment_scores(
    predicted_segments: list[tuple[float, float]],
    reference_segments: list[tuple[float, float]],
    tolerance: float,
) -> tuple[int, int, int]:
    """Calculate true positives, false positives, and false negatives.

    Args:
        predicted_segments: List of (start_time, end_time) tuples for predicted
            segments.
        reference_segments: List of (start_time, end_time) tuples for reference
            segments.
        tolerance: Time tolerance for matching segments.

    Returns:
        tp (int): Number of true positives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """
    total_predicted = len(predicted_segments)
    total_reference = len(reference_segments)

    if total_reference == 0:
        # No reference segments, all predictions are false positives
        return 0, total_predicted, 0

    tp = 0

    ref_array = np.array(reference_segments)
    pred_array = np.zeros_like(ref_array)

    for pred_start, pred_end in predicted_segments:
        pred_array[:, 0] = pred_start
        pred_array[:, 1] = pred_end

        diffs = np.abs(ref_array - pred_array)
        matches = np.all(diffs <= tolerance, axis=1)
        tp += 1 if np.any(matches) else 0

    fp = total_predicted - tp
    fn = total_reference - tp

    return tp, fp, fn


def f1_score_set(
    predicted_segments: list[list[tuple[float, float]]],
    reference_segments: list[list[tuple[float, float]]],
    tolerance: float,
) -> float:
    """Calculate the F1 score over a set of predicted and reference segments.

    Args:
        predicted_segments: List of lists of (start_time, end_time) tuples for
            predicted segments.
        reference_segments: List of lists of (start_time, end_time) tuples for
            reference segments.
        tolerance: Time tolerance for matching segments.

    Returns:
        f1_score (float): The F1 score calculated over the entire set.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    if len(predicted_segments) != len(reference_segments):
        msg = "Predicted and reference segments lists must have the same length"
        raise ValueError(msg)

    for pred_segs, ref_segs in zip(predicted_segments, reference_segments, strict=True):
        tp, fp, fn = segment_scores(pred_segs, ref_segs, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp == 0 or total_tp + total_fn == 0:
        return 0.0

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def dataset_to_segments(
    predictions: list[np.ndarray],
    time_start: float,
    time_step: float,
    on_threshold: float = 0.5,
    off_threshold: float | None = None,
    min_duration_off: float | None = None,
    min_duration_on: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Convert a dataset of predictions to segments.

    Args:
        predictions: List of numpy arrays containing model predictions.
        time_start: Center time for the first frame.
        time_step: Time difference between consecutive frames.
        on_threshold: Values above this threshold are considered "on".
        off_threshold: Values below this threshold are considered "off".
        min_duration_off: Minimum duration (in seconds) for "off" segments.
        min_duration_on: Minimum duration (in seconds) for "on" segments.
    Returns:
        all_segments (list of lists): List containing lists of (start_time,
            end_time) tuples for each clip in the dataset.
    """
    all_segments = []

    for prediction in predictions:
        binary_array = binarise(
            prediction,
            on_threshold=on_threshold,
            off_threshold=off_threshold,
            min_duration_off=min_duration_off,
            min_duration_on=min_duration_on,
        )
        segments = segment_times(binary_array, time_start, time_step)
        all_segments.append(segments)

    return all_segments


class SegmentEvaluator:
    """Class for evaluating segment predictions against references.
    The intent is for this to also be used in threshold optimisation.
    """

    def __init__(
        self,
        prediction_set: dict[str, np.ndarray],
        reference_set: dict[str, list[tuple[float, float]]],
        time_start: float,
        time_step: float,
        tolerance: float,
        threshold_on: float = 0.5,
        threshold_off: float | None = None,
        min_duration_off: float | None = None,
        min_duration_on: float | None = None,
    ):
        """Initialise the SegmentEvaluator.

        Args:
            prediction_set: Dictionary of numpy arrays containing model predictions.
            reference_set: Dictionary of lists of (start_time, end_time) tuples for
                reference segments.
            time_start: Center time for the first frame.
            time_step: Time difference between consecutive frames.
            tolerance: Time tolerance for matching segments.

        Optional Args:
            threshold_on: Values above this threshold are considered "on".
            threshold_off: Values below this threshold are considered "off".
            min_duration_off: Minimum duration (in seconds) for "off" segments.
            min_duration_on: Minimum duration (in seconds) for "on" segments.

        The optional arguments can be overridden later using set_parameters().
        """
        self.keys = []
        self.predictions = []
        self.references = []

        for key, prediction in prediction_set.items():
            if key in reference_set:
                self.keys.append(key)
                self.predictions.append(prediction)
                self.references.append(reference_set[key])
            else:
                msg = f"Key {key} not found in reference set"
                raise ValueError(msg)

        self.time_start = time_start
        self.time_step = time_step
        self.tolerance = tolerance

        self.main_threshold_on = threshold_on
        self.main_threshold_off = threshold_off
        self.main_min_duration_off = min_duration_off
        self.main_min_duration_on = min_duration_on

    def set_parameters(
        self,
        threshold_on: float | str = "no_change",
        threshold_off: float | None | str = "no_change",
        min_duration_off: float | None | str = "no_change",
        min_duration_on: float | None | str = "no_change",
    ) -> None:
        """Set the threshold and duration parameters.

        Args:
            threshold_on: New on threshold value, or "no_change" to keep
                current value. This must be a float.
                The default is "no_change".
            threshold_off: New off threshold value, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
            min_duration_off: New minimum off duration, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
            min_duration_on: New minimum on duration, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
        """
        if isinstance(threshold_on, str):
            if threshold_on != "no_change":
                msg = "threshold_on must be a float or not specified"
                raise ValueError(msg)
        else:
            self.main_threshold_on = threshold_on

        if isinstance(threshold_off, str):
            if threshold_off != "no_change":
                msg = "threshold_off must be a float, None, or not specified"
                raise ValueError(msg)
        else:
            self.main_threshold_off = threshold_off

        if isinstance(min_duration_off, str):
            if min_duration_off != "no_change":
                msg = "min_duration_off must be a float, None, or not specified"
                raise ValueError(msg)
        else:
            self.main_min_duration_off = min_duration_off

        if isinstance(min_duration_on, str):
            if min_duration_on != "no_change":
                msg = "min_duration_on must be a float, None, or not specified"
                raise ValueError(msg)
        else:
            self.main_min_duration_on = min_duration_on

    def get_parameters(self) -> dict[str, float | None]:
        """Get the current threshold and duration parameters.

        Returns:
            params (dict): Dictionary containing the current parameter values.
        """
        return {
            "threshold_on": self.main_threshold_on,
            "threshold_off": self.main_threshold_off,
            "min_duration_off": self.main_min_duration_off,
            "min_duration_on": self.main_min_duration_on,
        }

    def generate_segments(
        self,
        as_list: bool = False,
    ) -> dict[str, list[tuple[float, float]]] | list[list[tuple[float, float]]]:
        """Generate segments using the current threshold parameters.

        Returns:
            segments_dict (dict or list): Dictionary mapping keys to lists of
                (start_time, end_time) tuples for each sample.
        """
        segments = dataset_to_segments(
            self.predictions,
            self.time_start,
            self.time_step,
            on_threshold=self.main_threshold_on,
            off_threshold=self.main_threshold_off,
            min_duration_off=self.main_min_duration_off,
            min_duration_on=self.main_min_duration_on,
        )

        if as_list:
            return segments
        return dict(zip(self.keys, segments, strict=True))

    def f1_score(self) -> float:
        """Calculate the F1 score using the current threshold parameters.

        Returns:
            f1_score (float): The F1 score calculated over the entire set.
        """
        predicted_segments = self.generate_segments(as_list=True)
        assert isinstance(predicted_segments, list), "Unreachable type error"

        reference_segments = self.references

        return f1_score_set(predicted_segments, reference_segments, self.tolerance)
