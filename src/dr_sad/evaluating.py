from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class EvaluationMetrics:
    """
    Container for speech activity detection evaluation metrics.

    - Detection error rate = (Total missed speech + Total false Alarms) / Total speech
    frames
    - False alarm rate = Total false alarms / Total non-speech frames
    - Missed speech rate = Total missed speech / Total speech frames
    - Frame-level accuracy = Total correct frames / Total frames
    - Detection cost function (DCF) = weighted {false alarm rate + missed speech rate}
    (Normalised to 1)
    - F1 score (for speech and non-speech classes) = 2 * TP / (2 * TP + FP + FN)
    """

    der: float
    false_alarm_rate: float
    missed_speech_rate: float
    frame_accuracy: float
    detection_cost_function: float
    f1_speech: float
    f1_nonspeech: float

    def __str__(self) -> str:
        """
        For pretty printing of evaluation metrics.

        Returns:
            Formatted string of evaluation metrics.
        """
        return (
            f"Evaluation Metrics:\n"
            f"  DER: {self.der:.2%}\n"
            f"  DCF: {self.detection_cost_function:.2%}\n"
            f"  Frame Accuracy: {self.frame_accuracy:.2%}\n"
            f"  False Alarm: {self.false_alarm_rate:.2%}\n"
            f"  Missed Speech: {self.missed_speech_rate:.2%}\n"
            f"  F1 Speech: {self.f1_speech:.2%}\n"
            f"  F1 Non-speech: {self.f1_nonspeech:.2%}\n"
        )

    def format_for_plot(self) -> str:
        """
        Format evaluation metrics for plotting with concise labels.

        Returns:
            Formatted string of evaluation metrics for plot annotations.
        """
        return (
            f"Metrics:\n"
            f"DER: {self.der:.2%}\n"
            f"DCF: {self.detection_cost_function:.2%}\n"
            f"Acc: {self.frame_accuracy:.2%}\n"
            f"FA: {self.false_alarm_rate:.2%}\n"
            f"Miss: {self.missed_speech_rate:.2%}\n"
            f"F1 Spk: {self.f1_speech:.2%}\n"
            f"F1 Non-spk: {self.f1_nonspeech:.2%}"
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary using dataclasses.asdict"""
        return asdict(self)


def apply_collar(
    annotations: np.ndarray,
    collar_frames: int,
) -> np.ndarray:
    """
    Create a collar mask around segment boundaries. Identifies index at which transition
    occurs. Adds a collar of specified size around this index (inclusive of the index).

    Args:
        annotations: Binary annotation labels (0s and 1s)
        collar_frames: Number of frames for the collar on each side of boundaries

    Returns:
        Binary mask with 1s indicating collar regions (to be excluded from eval)
    """
    if collar_frames == 0:
        return np.zeros_like(annotations)

    if collar_frames < 0:
        err_msg = (
            f"collar_frames must be non-negative, did you mean {collar_frames * -1}?"
        )
        raise ValueError(err_msg)

    # Find boundaries (transitions between speech and non-speech)
    # Boundaries occur where the difference between adjacent frames is non-zero
    boundaries = np.abs(np.diff(annotations.astype(int)))
    # Diff produces an array one element shorter than input
    boundaries = np.concatenate([[0], boundaries])

    # Dilate boundaries to create collar regions
    # This expands each boundary by collar_frames on each side
    structure = np.ones(2 * collar_frames + 1)
    return binary_dilation(boundaries, structure=structure).astype(float)


def detection_error_rate(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    collar_frames: int = 0,
) -> dict[str, float]:
    """
    Compute Detection Error Rate (DER) metrics.

    Args:
        ground_truth: Binary ground truth mask (0 or 1)
        predictions: Model predictions (continuous values or binary)
        threshold: Threshold to binarise predictions (default: 0.5)
        collar_frames: Number of frames to ignore around speech boundaries

    Returns:
        Dictionary containing:
            - false_alarm_rate: False alarm rate
            - missed_speech_rate: Missed speech rate
            - der: Total detection error rate
            - collar_frames: Collar size used
    """
    # Binarise predictions
    pred_binary = (predictions > threshold).astype(float)

    # Ensure same length
    min_len = min(len(ground_truth), len(pred_binary))
    gt = ground_truth[:min_len]
    pred = pred_binary[:min_len]

    # Apply collar to ground truth if specified
    if collar_frames > 0:
        # Create collar mask (regions to ignore around boundaries)
        collar_mask = apply_collar(gt, collar_frames)
        # Evaluation mask: 0 in collar regions, 1 elsewhere
        eval_mask = 1 - collar_mask
    else:
        eval_mask = np.ones_like(gt)

    # Calculate errors only in evaluated regions
    false_alarm = np.sum(((pred == 1) & (gt == 0)) * eval_mask)
    missed_speech = np.sum(((pred == 0) & (gt == 1)) * eval_mask)

    # Total speech and non-speech frames (excluding collar)
    total_speech_frames = np.sum(gt * eval_mask)
    total_nonspeech_frames = np.sum((1 - gt) * eval_mask)

    # DER = (False Alarm + Missed Speech) / Total Speech Time
    if total_speech_frames > 0:
        der = (false_alarm + missed_speech) / total_speech_frames
        missed_speech_rate = missed_speech / total_speech_frames
    else:
        der = 0.0
        missed_speech_rate = 0.0

    # False alarm rate = proportion of non-speech incorrectly classified as speech
    if total_nonspeech_frames > 0:
        false_alarm_rate = false_alarm / total_nonspeech_frames
    else:
        false_alarm_rate = 0.0

    return {
        "der": der,
        "false_alarm_rate": false_alarm_rate,
        "missed_speech_rate": missed_speech_rate,
        "collar_frames": collar_frames,
    }


def detection_cost_function(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    collar_frames: int = 0,
) -> float:
    """
    Compute Detection Cost Function (DCF).

    Args:
        ground_truth: Binary ground truth mask (0 or 1)
        predictions: Model predictions (continuous values or binary)
        threshold: Threshold to binarise predictions (default: 0.5)
        collar_frames: Number of frames to ignore around speech boundaries

    Returns:
        Detection Cost Function value as a float.
    """
    der_metrics = detection_error_rate(
        ground_truth,
        predictions,
        threshold=threshold,
        collar_frames=collar_frames,
    )

    # Weights for DCF calculation
    C_miss = 0.75  # Cost of missed detection - from pyannote implementation
    C_fa = 0.25  # Cost of false alarm - from pyannote implementation

    return (
        C_miss * der_metrics["missed_speech_rate"]
        + C_fa * der_metrics["false_alarm_rate"]
    )


def frame_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    collar_frames: int = 0,
) -> float:
    """
    Compute frame-level accuracy.

    Args:
        predictions: Model predictions (continuous values)
        ground_truth: Binary ground truth mask (0 or 1)
        threshold: Threshold to binarize predictions (default: 0.5)
        collar_frames: Number of frames to ignore around speech boundaries (default: 0)

    Returns:
        Frame-level accuracy as a float.
    """
    # Binarise predictions
    pred_binary = (predictions > threshold).astype(float)

    # Ensure same length
    min_len = min(len(ground_truth), len(pred_binary))
    gt = ground_truth[:min_len]
    pred = pred_binary[:min_len]

    correct_frames = np.sum(pred == gt)  # can return Any, so cast to float
    total_frames = len(gt)

    if collar_frames > 0:
        # Create collar mask (regions to ignore around boundaries)
        collar_mask = apply_collar(gt, collar_frames)
        # Evaluation mask: 0 in collar regions, 1 elsewhere
        eval_mask = 1 - collar_mask

        correct_frames = np.sum(((pred == gt) * eval_mask).astype(float))
        total_frames = np.sum(eval_mask)

    return float(correct_frames / total_frames) if total_frames > 0 else 0.0


class SpeechDetectionEvaluator:
    def __init__(
        self, collar_frames: int = 0, detection_threshold: float = 0.5
    ) -> None:
        """
        Initialize the evaluator with optional collar size.

        Args:
            collar_frames: Number of frames to ignore around speech boundaries
        """
        self.collar_frames = collar_frames
        self.detection_threshold = detection_threshold

    def calculate_base_metrics(self, ground_truth, predictions) -> dict[str, float]:
        """
        Calculate base values for frame-wise metrics. This is a utility function which
        will be used in calculating other metrics.

        Args:
            ground_truth: Binary ground truth mask
            predictions: Model predictions

        Returns:
            Total frames
            Real positives
            Real negatives
            False Positives
            False Negatives
            True Positives
            True Negatives
        """
        pred_binary = (predictions > self.detection_threshold).astype(float)

        if self.collar_frames > 0:
            collar_mask = apply_collar(ground_truth, self.collar_frames)
            eval_mask = 1 - collar_mask
        else:
            eval_mask = np.ones_like(ground_truth)

        # Apply eval_mask to all calculations
        true_positives = np.sum(
            (pred_binary == 1) & (ground_truth == 1) & (eval_mask == 1)
        )
        true_negatives = np.sum(
            (pred_binary == 0) & (ground_truth == 0) & (eval_mask == 1)
        )
        false_positives = np.sum(
            (pred_binary == 1) & (ground_truth == 0) & (eval_mask == 1)
        )
        false_negatives = np.sum(
            (pred_binary == 0) & (ground_truth == 1) & (eval_mask == 1)
        )

        real_positives = np.sum((ground_truth == 1) & (eval_mask == 1))
        real_negatives = np.sum((ground_truth == 0) & (eval_mask == 1))
        total_frames = np.sum(eval_mask)

        return {
            "total_frames": total_frames,
            "real_positives": real_positives,
            "real_negatives": real_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
        }

    def evaluate(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.

        Args:
            ground_truth: Binary ground truth mask
            predictions: Model predictions

        Returns:
            EvaluationMetrics object
        """
        base_metrics = self.calculate_base_metrics(ground_truth, predictions)

        false_alarms = base_metrics["false_positives"]
        missed_speech = base_metrics["false_negatives"]
        total_speech_frames = base_metrics["real_positives"]
        total_nonspeech_frames = base_metrics["real_negatives"]
        total_frames = base_metrics["total_frames"]
        true_positives = base_metrics["true_positives"]
        true_negatives = base_metrics["true_negatives"]

        der = (
            (false_alarms + missed_speech) / total_speech_frames
            if total_speech_frames > 0
            else 0.0
        )
        false_alarm_rate = (
            false_alarms / total_nonspeech_frames if total_nonspeech_frames > 0 else 0.0
        )
        missed_speech_rate = (
            missed_speech / total_speech_frames if total_speech_frames > 0 else 0.0
        )
        frame_accuracy = (
            (true_positives + true_negatives) / total_frames
            if total_frames > 0
            else 0.0
        )
        detection_cost_function = 0.75 * missed_speech_rate + 0.25 * false_alarm_rate
        f1_speech = (
            (2 * true_positives) / ((2 * true_positives) + false_alarms + missed_speech)
            if ((2 * true_positives) + false_alarms + missed_speech) > 0
            else 0.0
        )
        f1_nonspeech = (
            (2 * true_negatives) / ((2 * true_negatives) + missed_speech + false_alarms)
            if ((2 * true_negatives) + missed_speech + false_alarms) > 0
            else 0.0
        )

        return EvaluationMetrics(
            der=float(der),
            false_alarm_rate=float(false_alarm_rate),
            missed_speech_rate=float(missed_speech_rate),
            frame_accuracy=float(frame_accuracy),
            detection_cost_function=float(detection_cost_function),
            f1_speech=float(f1_speech),
            f1_nonspeech=float(f1_nonspeech),
        )
