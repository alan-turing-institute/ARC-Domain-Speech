import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
from scipy import signal as sp_signal

from dr_sad.evaluating import EvaluationMetrics, SpeechDetectionEvaluator


def _load_audio_and_annotations(
    file_id: str,
    data_dir: str = "data/callhome",
) -> tuple[np.ndarray, int, list[tuple[float, float]]]:
    """Load audio file and corresponding RTTM annotations."""
    data_path = Path(data_dir)

    # Load audio file
    audio_path = data_path / "flac" / f"{file_id}.flac"
    audio, sample_rate = soundfile.read(audio_path)

    # Load RTTM annotations
    rttm_path = data_path / "rttm" / f"{file_id}.rttm"
    speech_segments = []

    with open(rttm_path) as f:
        for line in f:
            if line.strip() and line.startswith("SPEAKER"):
                parts = line.strip().split()
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speech_segments.append((start_time, end_time))

    return audio, sample_rate, speech_segments


def _create_ground_truth_mask(
    audio_length: int,
    sample_rate: int,
    speech_segments: list[tuple[float, float]],
) -> np.ndarray:
    """Create a binary mask for ground truth speech activity."""
    mask = np.zeros(audio_length)

    for start_time, end_time in speech_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(audio_length, end_sample)
        mask[start_sample:end_sample] = 1.0

    return mask


def _downsample_to_prediction_frames(
    signal: np.ndarray,
    prediction_length: int,
    is_binary: bool = False,
) -> np.ndarray:
    """
    Downsample signal to match prediction frame rate.

    Args:
        signal: Input signal to downsample
        prediction_length: Target length after downsampling
        is_binary: If True, uses nearest-neighbor resampling to preserve binary values.
                   If False, uses spectral interpolation for smooth resampling.

    Returns:
        Downsampled signal
    """

    if is_binary:
        # For binary signals, use nearest-neighbor interpolation
        # This preserves the binary nature without introducing fractional values
        indices = np.round(np.linspace(0, len(signal) - 1, prediction_length)).astype(
            int
        )
        downsampled = signal[indices].astype(float)
    else:
        # For continuous signals (audio), use spectral interpolation
        # This provides smooth, anti-aliased downsampling
        downsampled = sp_signal.resample(signal, prediction_length)

    return downsampled


def plot_analysis(
    file_id: str,
    audio: np.ndarray,
    sample_rate: int,
    ground_truth_mask: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    evaluation_metrics: EvaluationMetrics | None = None,
) -> None:
    """
    Create a single plot with audio background, ground truth, and predictions overlaid.

    Args:
        file_id: ID of the file being analyzed
        audio: Raw audio samples
        sample_rate: Audio sample rate
        ground_truth_mask: Binary mask for ground truth speech activity
        predictions: Model predictions
        output_dir: Directory to save plots
        evaluation_metrics: EvaluationMetrics object for displaying metrics on plot
    """

    # Convert predictions to numpy if needed
    if torch.is_tensor(predictions):
        pred_probs = predictions.detach().cpu().numpy().squeeze()
    else:
        pred_probs = predictions.squeeze()

    # Downsample both audio and ground truth to match prediction length
    pred_length = len(pred_probs)
    # Create single time axis for all signals
    audio_duration = len(audio) / sample_rate
    pred_times = np.linspace(0, audio_duration, pred_length)

    # Downsample ground truth
    gt_downsampled = _downsample_to_prediction_frames(
        ground_truth_mask, pred_length, is_binary=True
    )

    # Downsample audio
    audio_downsampled = _downsample_to_prediction_frames(audio, pred_length)
    # Normalize audio to [-0.5, 0.5] then shift to [0, 1] for plotting
    if np.max(np.abs(audio_downsampled)) > 0:
        audio_normalised = audio_downsampled / (2 * np.max(np.abs(audio_downsampled)))
        audio_normalised = audio_normalised + 0.5
    else:
        audio_normalised = np.ones_like(audio_downsampled) * 0.5

    # Create figure
    _, ax = plt.subplots(figsize=(16, 6))

    # Plot downsampled audio waveform in background
    ax.plot(
        pred_times,
        audio_normalised,
        color="lightgray",
        alpha=0.5,
        linewidth=0.5,
        label="Audio waveform",
    )
    # Plot ground truth as fill_between
    ax.fill_between(
        pred_times, 0, gt_downsampled, color="green", alpha=0.3, label="Ground Truth"
    )

    # Plot predictions
    ax.plot(pred_times, pred_probs, label="Predictions", color="red", alpha=0.7)

    # Labels and formatting
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speech Activity")
    ax.set_title(f"Speech Activity Detection: {file_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add DER metrics text box if provided
    if evaluation_metrics:
        metrics_text = evaluation_metrics.format_for_plot()
        # Add text box in upper right corner
        ax.text(
            0.98,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{file_id}_analysis.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_file(
    file_id: str,
    predictions: np.ndarray,
    model_metadata: dict[str, Any],
    evaluator: SpeechDetectionEvaluator,
    output_dir: None | Path = None,
    plot_figures: bool = False,
) -> EvaluationMetrics:
    """Analyze a single file with audio, ground truth, and predictions.

    Args:
        file_id: ID of the file to analyze
        predictions: Model predictions for this file
        model_metadata: Metadata dictionary for the model
        output_dir: Directory to save plots (only required if plot_figures is True)
        evaluator: SpeechDetectionEvaluator instance for metric calculations
        plot_figures: Whether to generate and save plots
    returns:
        EvaluationMetrics object with computed metrics, or None if evaluator is None
    """

    # check plotting arguments are consistent
    if output_dir is None and plot_figures:
        err_msg = "output_dir must be provided if plot_figures is True."
        raise ValueError(err_msg)

    if output_dir is not None and not plot_figures:
        warnings.warn(
            "output_dir provided but plot_figures is False."
            " output_dir will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    audio, sample_rate, speech_segments = _load_audio_and_annotations(file_id)

    # Create ground truth mask
    ground_truth_mask = _create_ground_truth_mask(
        len(audio), sample_rate, speech_segments
    )
    frame_rate_hz = model_metadata["frame_rate_hz"]
    audio_duration_sec = len(audio) / sample_rate
    actual_num_frames = int(audio_duration_sec * frame_rate_hz)

    # get only the valid portion of predictions
    signal_predictions = predictions[:actual_num_frames]

    # Downsample ground truth to match predictions
    gt_downsampled = _downsample_to_prediction_frames(
        ground_truth_mask, len(signal_predictions), is_binary=True
    )
    # Compute DER metrics
    evaluation_metrics = evaluator.evaluate(
        predictions=signal_predictions, ground_truth=gt_downsampled
    )

    if plot_figures and output_dir is not None:
        # Create visualization
        plot_analysis(
            file_id,
            audio,
            sample_rate,
            ground_truth_mask,
            signal_predictions,
            output_dir,
            evaluation_metrics=evaluation_metrics,
        )

    return evaluation_metrics
