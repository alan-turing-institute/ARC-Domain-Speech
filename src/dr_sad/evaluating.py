from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class EvaluationMetrics:
    """Container for speech activity detection evaluation metrics."""

    der: float
    """Detection error rate"""

    false_alarm_rate: float
    """False alarm rate"""

    missed_speech_rate: float
    """Missed speech rate"""

    accuracy: float
    """Frame-level accuracy"""

    detection_cost_function: float
    """Detection cost function (DCF)"""

    def __str__(self) -> str:
        return (
            f"Evaluation Metrics:\n"
            f"  DER: {self.der:.2%}\n"
            f"  DCF: {self.detection_cost_function:.2%}\n"
            f"  Frame Accuracy: {self.accuracy:.2%}\n"
            f"  False Alarm: {self.false_alarm_rate:.2%}\n"
            f"  Missed Speech: {self.missed_speech_rate:.2%}\n"
        )

    def format_for_plot(self) -> str:
        return (
            f"Metrics:\n"
            f"DER: {self.der:.2%}\n"
            f"DCF: {self.detection_cost_function:.2%}\n"
            f"Acc: {self.accuracy:.2%}\n"
            f"FA: {self.false_alarm_rate:.2%}\n"
            f"Miss: {self.missed_speech_rate:.2%}"
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary using dataclasses.asdict"""
        return asdict(self)


def apply_collar(
    mask: np.ndarray,
    collar_frames: int,
) -> np.ndarray:
    """
    Create a collar mask around segment boundaries.

    The collar removes frames around boundaries from evaluation (half the collar
    duration before the boundary, half after). This accounts for uncertainty in
    exact boundary detection.

    Args:
        mask: Binary mask (0s and 1s)
        collar_frames: Number of frames for the collar on each side of boundaries

    Returns:
        Binary mask with 1s indicating collar regions (to be excluded from eval)
    """
    if collar_frames == 0:
        return np.zeros_like(mask)

    if collar_frames < 0:
        err_msg = (
            f"collar_frames must be non-negative, did you mean {collar_frames * -1}?"
        )
        raise ValueError(err_msg)

    # Find boundaries (transitions between speech and non-speech)
    # Boundaries occur where the difference between adjacent frames is non-zero
    boundaries = np.abs(np.diff(mask.astype(int)))
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

    def calculate_detection_error_rate(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate detection error rate metrics.

        Args:
            ground_truth: Binary ground truth mask
            predictions: Model predictions

        Returns:
            Dictionary with DER metrics.
        """
        return detection_error_rate(
            ground_truth,
            predictions,
            threshold=self.detection_threshold,
            collar_frames=self.collar_frames,
        )

    def calculate_frame_accuracy(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """
        Calculate frame-level accuracy.

        Args:
            ground_truth: Binary ground truth mask
            predictions: Model predictions

        Returns:
            Frame-level accuracy as a float.
        """
        return frame_accuracy(
            predictions,
            ground_truth,
            threshold=self.detection_threshold,
            collar_frames=self.collar_frames,
        )

    def calculate_detection_cost_function(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """
        Calculate detection cost function (DCF).

        Args:
            ground_truth: Binary ground truth mask
            predictions: Model predictions

        Returns:
            DCF as a float.
        """
        return detection_cost_function(
            ground_truth,
            predictions,
            threshold=self.detection_threshold,
            collar_frames=self.collar_frames,
        )

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
        der_results = self.calculate_detection_error_rate(ground_truth, predictions)
        accuracy = self.calculate_frame_accuracy(ground_truth, predictions)
        dcf = self.calculate_detection_cost_function(ground_truth, predictions)

        return EvaluationMetrics(
            der=der_results["der"],
            false_alarm_rate=der_results["false_alarm_rate"],
            missed_speech_rate=der_results["missed_speech_rate"],
            accuracy=accuracy,
            detection_cost_function=dcf,
        )
