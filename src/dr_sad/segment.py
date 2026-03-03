"""Tools for generating and handling segments."""

import numpy as np
from scipy import optimize


def binarise(
    input: np.ndarray,
    speech_threshold: float = 0.5,
    gap_threshold: float | None = None,
    min_duration_off: float | None = None,
    min_duration_on: float | None = None,
) -> np.ndarray:
    """Binarise an input array based on thresholds and minimum durations.

    Args:
        input: 1D numpy array of float values to be binarised.
        speech_threshold: Center threshold for speech detection.
        gap_threshold: Gap around speech_threshold. If specified, values above
            speech_threshold + gap_threshold/2 are "on", values below
            speech_threshold - gap_threshold/2 are "off". If None, uses simple
            threshold at speech_threshold.
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

    if gap_threshold is None:
        on_threshold = speech_threshold
        off_threshold = None
        binary_array = input >= speech_threshold
    else:
        if gap_threshold < 0:
            msg = "gap_threshold must be non-negative"
            raise ValueError(msg)
        on_threshold = speech_threshold + gap_threshold / 2
        off_threshold = speech_threshold - gap_threshold / 2

        binary_array = np.zeros_like(input, dtype=bool)
        current = input[0] >= speech_threshold
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
    sample_weights: list[float] | None = None,
) -> float:
    """Calculate the F1 score over a set of predicted and reference segments.
    The score is calculated by summing true positives, false positives, and false
    negatives across the entire set. If a weighting is provided, these values are
    weighted individually before summing.

    Args:
        predicted_segments: List of lists of (start_time, end_time) tuples for
            predicted segments.
        reference_segments: List of lists of (start_time, end_time) tuples for
            reference segments.
        tolerance: Time tolerance for matching segments.
        sample_weights: Optional list of weights for each sample. If provided,
            the F1 score will be calculated as a weighted average.

    Returns:
        f1_score (float): The F1 score calculated over the entire set.
    """
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    if len(predicted_segments) != len(reference_segments):
        msg = "Predicted and reference segments lists must have the same length"
        raise ValueError(msg)
    if sample_weights is not None and len(sample_weights) != len(predicted_segments):
        msg = (
            "Length of sample_weights must match length of predicted_segments "
            "if provided"
        )
        raise ValueError(msg)

    for n in range(len(predicted_segments)):
        tp, fp, fn = segment_scores(
            predicted_segments[n], reference_segments[n], tolerance
        )
        factor = 1.0 if sample_weights is None else sample_weights[n]
        total_tp += tp * factor
        total_fp += fp * factor
        total_fn += fn * factor

    if np.abs(total_tp + total_fp) < 1e-9 or np.abs(total_tp + total_fn) < 1e-9:
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
    speech_threshold: float = 0.5,
    gap_threshold: float | None = None,
    min_duration_off: float | None = None,
    min_duration_on: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Convert a dataset of predictions to segments.

    Args:
        predictions: List of numpy arrays containing model predictions.
        time_start: Center time for the first frame.
        time_step: Time difference between consecutive frames.
        speech_threshold: Center threshold for speech detection.
        gap_threshold: Gap around speech_threshold for hysteresis.
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
            speech_threshold=speech_threshold,
            gap_threshold=gap_threshold,
            min_duration_off=min_duration_off,
            min_duration_on=min_duration_on,
        )
        segments = segment_times(binary_array, time_start, time_step)
        all_segments.append(segments)

    return all_segments


class DiffEvolOptimizer:
    """Class for optimizing parameters using Differential Evolution."""

    def __init__(
        self,
        predictions: list[np.ndarray],
        references: list[list[tuple[float, float]]],
        sample_weights: list[float] | None,
        start_parameters: dict[str, float | None],
        optimise_parameters: list[str],
        time_start: float,
        time_step: float,
        tolerance: float,
    ):
        self.predictions = predictions
        self.references = references
        self.sample_weights = sample_weights
        self.parameters = start_parameters
        self.optimise_parameters = optimise_parameters
        self.time_start = time_start
        self.time_step = time_step
        self.tolerance = tolerance

        for param_name in [
            "speech_threshold",
            "gap_threshold",
            "min_duration_off",
            "min_duration_on",
        ]:
            if param_name not in self.parameters:
                msg = f"Parameter {param_name} not found in start_parameters"
                raise ValueError(msg)

        if self.parameters["speech_threshold"] is None:
            # This probably shouldn't be reachable
            msg = "speech_threshold must be specified in start_parameters"
            raise ValueError(msg)

    def __call__(self, params: list[float]) -> float:
        for name, value in zip(self.optimise_parameters, params, strict=True):
            self.parameters[name] = float(value)

        segments = dataset_to_segments(
            self.predictions,
            self.time_start,
            self.time_step,
            speech_threshold=self.parameters["speech_threshold"],  # type: ignore[arg-type]
            gap_threshold=self.parameters["gap_threshold"],
            min_duration_off=self.parameters["min_duration_off"],
            min_duration_on=self.parameters["min_duration_on"],
        )

        return -f1_score_set(
            segments, self.references, self.tolerance, self.sample_weights
        )


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
        sample_weighting: dict[str, float] | None = None,
        speech_threshold: float = 0.5,
        gap_threshold: float | None = None,
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
            sample_weighting: Optional dictionary mapping keys to weights for
                calculating weighted F1 scores.

        Optional Args:
            speech_threshold: Center threshold for speech detection.
            gap_threshold: Gap around speech_threshold for hysteresis.
            min_duration_off: Minimum duration (in seconds) for "off" segments.
            min_duration_on: Minimum duration (in seconds) for "on" segments.

        The optional arguments can be overridden later using set_parameters().
        """
        self.keys = []
        self.predictions = []
        self.references = []
        self.sample_weights: list[float] | None = (
            [] if sample_weighting is not None else None
        )

        for key, prediction in prediction_set.items():
            if key in reference_set:
                self.keys.append(key)
                self.predictions.append(prediction)
                self.references.append(reference_set[key])
                if isinstance(sample_weighting, dict):
                    if key not in sample_weighting:
                        msg = f"Key {key} not found in sample_weighting"
                        raise ValueError(msg)
                    if not isinstance(self.sample_weights, list):
                        msg = "Unreachable sample weights type error"
                        raise RuntimeError(msg)
                    self.sample_weights.append(sample_weighting[key])
            else:
                msg = f"Key {key} not found in reference set"
                raise ValueError(msg)

        self.time_start = time_start
        self.time_step = time_step
        self.tolerance = tolerance

        self.main_speech_threshold = speech_threshold
        self.main_gap_threshold = gap_threshold
        self.main_min_duration_off = min_duration_off
        self.main_min_duration_on = min_duration_on

    def set_parameters(
        self,
        speech_threshold: float | str = "no_change",
        gap_threshold: float | None | str = "no_change",
        min_duration_off: float | None | str = "no_change",
        min_duration_on: float | None | str = "no_change",
    ) -> None:
        """Set the threshold and duration parameters.

        Args:
            speech_threshold: New speech threshold value, or "no_change" to keep
                current value. This must be a float.
                The default is "no_change".
            gap_threshold: New gap threshold value, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
            min_duration_off: New minimum off duration, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
            min_duration_on: New minimum on duration, or "no_change" to keep
                current value. This can be a float or None.
                The default is "no_change".
        """
        if isinstance(speech_threshold, str):
            if speech_threshold != "no_change":
                msg = "speech_threshold must be a float or not specified"
                raise ValueError(msg)
        else:
            self.main_speech_threshold = float(speech_threshold)

        if isinstance(gap_threshold, str):
            if gap_threshold != "no_change":
                msg = "gap_threshold must be a float, None, or not specified"
                raise ValueError(msg)
        elif gap_threshold is not None:
            self.main_gap_threshold = float(gap_threshold)
        else:
            self.main_gap_threshold = None

        if isinstance(min_duration_off, str):
            if min_duration_off != "no_change":
                msg = "min_duration_off must be a float, None, or not specified"
                raise ValueError(msg)
        elif min_duration_off is not None:
            self.main_min_duration_off = float(min_duration_off)
        else:
            self.main_min_duration_off = None

        if isinstance(min_duration_on, str):
            if min_duration_on != "no_change":
                msg = "min_duration_on must be a float, None, or not specified"
                raise ValueError(msg)
        elif min_duration_on is not None:
            self.main_min_duration_on = float(min_duration_on)
        else:
            self.main_min_duration_on = None

    def get_parameters(self) -> dict[str, float | None]:
        """Get the current threshold and duration parameters.

        Returns:
            params (dict): Dictionary containing the current parameter values.
        """
        return {
            "speech_threshold": self.main_speech_threshold,
            "gap_threshold": self.main_gap_threshold,
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
            speech_threshold=self.main_speech_threshold,
            gap_threshold=self.main_gap_threshold,
            min_duration_off=self.main_min_duration_off,
            min_duration_on=self.main_min_duration_on,
        )

        if as_list:
            return segments
        return dict(zip(self.keys, segments, strict=True))

    def f1_score(self) -> float:
        """Calculate the F1 score using the current threshold parameters.
        This is calculated by generating the true positives, false positives, and false
        negatives across the entire set and then calculating the F1 score from these
        totals. If weighing is provided counts are weighted before summing.

        Returns:
            f1_score (float): The F1 score calculated over the entire set.
        """
        predicted_segments = self.generate_segments(as_list=True)
        assert isinstance(predicted_segments, list), "Unreachable type error"

        reference_segments = self.references

        return f1_score_set(
            predicted_segments,
            reference_segments,
            self.tolerance,
            sample_weights=self.sample_weights,
        )

    def optimise_diff_evol(
        self,
        speech_threshold: bool = True,
        gap_threshold: bool = True,
        min_duration_off: bool = True,
        min_duration_on: bool = True,
        num_workers: int = -1,
        maxiter: int | None = None,
    ) -> optimize.OptimizeResult:
        """Optimise the threshold and duration parameters using Differential Evolution.

        Args:
            speech_threshold: Whether to optimise speech threshold.
            gap_threshold: Whether to optimise gap threshold.
            min_duration_off: Whether to optimise minimum off duration.
            min_duration_on: Whether to optimise minimum on duration.
            num_workers: Number of parallel workers to use (-1 uses all available).
            maxiter: Maximum number of iterations for the optimiser.
                (This is mostly included for testing purposes.)

        Returns:
            result (OptimizeResult): The result of the optimisation process.
        """
        param_names = []
        bounds = []

        if speech_threshold:
            param_names.append("speech_threshold")
            bounds.append((0.0, 1.0))

        if gap_threshold:
            param_names.append("gap_threshold")
            bounds.append((0.0, 1.0))

        if min_duration_off:
            param_names.append("min_duration_off")
            bounds.append((0.0, 5.0))

        if min_duration_on:
            param_names.append("min_duration_on")
            bounds.append((0.0, 5.0))

        if not param_names:
            msg = "No parameters selected for optimisation"
            raise ValueError(msg)

        optimiser = DiffEvolOptimizer(
            predictions=self.predictions,
            references=self.references,
            sample_weights=self.sample_weights,
            start_parameters=self.get_parameters(),
            optimise_parameters=param_names,
            time_start=self.time_start,
            time_step=self.time_step,
            tolerance=self.tolerance,
        )

        optimise_result = optimize.differential_evolution(
            optimiser,
            bounds=bounds,
            workers=num_workers,
            updating="deferred",
            maxiter=maxiter,
        )

        self.set_parameters(**dict(zip(param_names, optimise_result.x, strict=True)))

        if optimise_result.success:
            print("Differential Evolution optimisation successful.")
            print(f"Used {optimise_result.nfev} function evaluations.")
            print("Optimised parameters:")
            for name in param_names:
                value = self.get_parameters()[name]
                print(f"  {name}: {value}")
        else:
            print("Differential Evolution optimisation failed.")

        return optimise_result

    def optimise_parameters(
        self,
        speech_threshold: bool = True,
        gap_threshold: bool = True,
        min_duration_off: bool = True,
        min_duration_on: bool = True,
        optimise_method: str = "Powell",
        maxiter: int | None = None,
    ) -> optimize.OptimizeResult:
        """Optimise the threshold and duration parameters to maximise F1 score.

        Args:
            speech_threshold: Whether to optimise speech threshold.
            gap_threshold: Whether to optimise gap threshold.
            min_duration_off: Whether to optimise minimum off duration.
            min_duration_on: Whether to optimise minimum on duration.
            optimise_method: The optimisation method to use (default is 'Powell').
            maxiter: Maximum number of iterations for the optimiser.
                (This is mostly included for testing purposes.)

        Returns:
            result (OptimizeResult): The result of the optimisation process.
        """
        param_names = []
        initial_values = []
        bounds = []

        if speech_threshold:
            param_names.append("speech_threshold")
            initial_values.append(self.main_speech_threshold)
            bounds.append((0.0, 1.0))

        if gap_threshold:
            param_names.append("gap_threshold")
            initial_values.append(
                self.main_gap_threshold if self.main_gap_threshold is not None else 0.1
            )
            bounds.append((0.0, 1.0))

        if min_duration_off:
            param_names.append("min_duration_off")
            initial_values.append(
                self.main_min_duration_off
                if self.main_min_duration_off is not None
                else 0.2
            )
            bounds.append((0.0, 5.0))

        if min_duration_on:
            param_names.append("min_duration_on")
            initial_values.append(
                self.main_min_duration_on
                if self.main_min_duration_on is not None
                else 0.2
            )
            bounds.append((0.0, 5.0))

        if not param_names:
            msg = "No parameters selected for optimisation"
            raise ValueError(msg)

        def objective(params: list[float]) -> float:
            self.set_parameters(**dict(zip(param_names, params, strict=True)))
            f1 = self.f1_score()
            return -f1

        optimise_result = optimize.minimize(
            objective,
            initial_values,
            bounds=bounds,
            method=optimise_method,
            options={"maxiter": maxiter} if maxiter is not None else None,
        )

        self.set_parameters(**dict(zip(param_names, optimise_result.x, strict=True)))

        return optimise_result
