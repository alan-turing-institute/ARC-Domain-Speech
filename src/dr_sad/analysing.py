from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
from scipy import signal as sp_signal


def mode(x: np.ndarray) -> np.ndarray:
    """Return the mode of a 1D numpy array."""
    values, counts = np.unique(x, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]


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
    """Downsample signal to match prediction frame rate using vectorized operations."""

    if is_binary:
        # Use scipy's resample with nearest neighbor for binary
        downsampled: np.ndarray = sp_signal.resample(
            signal, prediction_length, window="boxcar"
        )
        # Re-binarize after resampling (threshold at 0.5)
        downsampled = (downsampled > 0.5).astype(float)
    else:
        # For continuous signals, scipy.signal.resample is efficient
        downsampled = sp_signal.resample(signal, prediction_length)

    return downsampled


def plot_analysis(
    file_id: str,
    audio: np.ndarray,
    sample_rate: int,
    ground_truth_mask: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path | None = None,
) -> None:
    """
    Create a single plot with audio background, ground truth, and predictions overlaid.

    Args:
        file_id: ID of the file being analyzed
        audio: Raw audio samples
        sample_rate: Audio sample rate
        ground_truth_mask: Binary mask for ground truth speech activity
        predictions: Model predictions
        output_dir: Directory to save plots (if None, uses .temp/)
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

    # Save plot
    if output_dir is None:
        output_dir = Path(".temp")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{file_id}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyse_file(
    file_id: str,
    predictions: torch.Tensor,
    model_metadata: dict[str, Any],
    output_dir: Path | None = None,
) -> None:
    """Analyze a single file with audio, ground truth, and predictions.

    Args:
        file_id: ID of the file to analyze
        predictions: Model predictions for this file
        model_metadata: Metadata dictionary for the model
        output_dir: Directory to save plots (if None, uses .temp/)
    """

    audio, sample_rate, speech_segments = _load_audio_and_annotations(file_id)

    # Create ground truth mask
    ground_truth_mask = _create_ground_truth_mask(
        len(audio), sample_rate, speech_segments
    )
    frame_rate_hz = model_metadata["frame_rate_hz"]
    audio_duration_sec = len(audio) / sample_rate
    actual_num_frames = int(audio_duration_sec * frame_rate_hz)
    predictions = predictions[:, :actual_num_frames]

    # Create visualization
    plot_analysis(
        file_id,
        audio,
        sample_rate,
        ground_truth_mask,
        predictions,
        output_dir,
    )
