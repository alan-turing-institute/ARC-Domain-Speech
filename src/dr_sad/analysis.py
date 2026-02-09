import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile
import torch
from scipy import signal as sp_signal

from dr_sad.annotation import speaking_map
from dr_sad.evaluating import EvaluationMetrics, SpeechDetectionEvaluator


def inverse_weightings_by_domain(
    file_ids: list[str], data_tbl_path: Path
) -> dict[str, dict[str, float] | dict[str, int]]:
    """
    Calculate inverse normalisation weightings for each domain based on the number of
    files in each domain.
    Less represented domains get higher weights.

    Args:
        file_ids: List of file IDs to be analyzed.
        data_tbl_path: Path to the data table CSV file containing file metadata,
            including domain information.

    Returns:
        Dictionary mapping file IDs to their corresponding inverse normalisation
        weightings.
    """

    # Load data table
    data_tbl = pd.read_csv(data_tbl_path, sep="\t")

    # Filter data table to only include rows corresponding to the provided file_ids
    filtered_data_tbl = data_tbl[data_tbl["file_id"].isin(file_ids)]

    # Count number of files in each domain
    domain_counts = filtered_data_tbl["domain"].value_counts().to_dict()

    # Calculate total number of files
    total_files = len(file_ids)

    # Calculate inverse weightings for each domain
    inverse_weights = {}
    for domain, count in domain_counts.items():
        inverse_weights[domain] = total_files / count

    # Normalize inverse weights so they sum to 1
    total_weight = sum(inverse_weights.values())
    normalized_weights = {
        domain: weight / total_weight for domain, weight in inverse_weights.items()
    }

    # Create file-level weights mapping
    file_weights = {}
    for file_id in file_ids:
        domain = filtered_data_tbl[filtered_data_tbl["file_id"] == file_id][
            "domain"
        ].values[0]
        file_weights[file_id] = normalized_weights[domain]

    return {
        "domain_weights": normalized_weights,
        "domain_counts": domain_counts,
        "file_weights": file_weights,
    }


def load_annotations(
    file_id: str,
    data_dir: Path | str,
) -> list[tuple[float, float]]:
    """Load RTTM annotations.

    Args:
        file_id (str): Identifier for the audio file.
        data_path (str | Path): Directory containing the RTTM files.

    Returns:
        speech_segments (list[tuple[float, float]]): List of speech segments as
            tuples of (start_time, end_time).
    """
    data_path = Path(data_dir) if isinstance(data_dir, str) else data_dir

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

    return speech_segments


def load_audio_and_annotations(
    file_id: str,
    data_dir: str | Path,
) -> tuple[np.ndarray, int, list[tuple[float, float]]]:
    """Load audio file and corresponding RTTM annotations.

    Args:
        file_id (str): Identifier for the audio file.
        data_path (str | Path): Directory containing the audio and RTTM files.

    Returns:
        audio (np.ndarray): Loaded audio samples.
        sample_rate (int): Sample rate of the audio file.
        speech_segments (list[tuple[float, float]]): List of speech segments as
            tuples of (start_time, end_time).
    """
    data_path = Path(data_dir) if isinstance(data_dir, str) else data_dir

    # Load audio file
    audio_path = data_path / "flac" / f"{file_id}.flac"
    audio, sample_rate = soundfile.read(audio_path)

    speech_segments = load_annotations(file_id, data_path)

    return audio, sample_rate, speech_segments


def downsample_to_prediction_frames(
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
    ground_truth_mask: np.ndarray,
    predictions: np.ndarray,
    timestamps: np.ndarray,
    output_dir: Path,
    evaluation_metrics: EvaluationMetrics | None = None,
) -> None:
    """
    Create a single plot with audio background, ground truth, and predictions overlaid.

    Args:
        file_id: ID of the file being analysed
        audio: Raw audio samples
        ground_truth_mask: Binary mask for ground truth speech activity
        predictions: Model predictions
        timestamps: Time values corresponding to prediction frames
        output_dir: Directory to save plots
        evaluation_metrics: EvaluationMetrics object for displaying metrics on plot
    """

    # Convert predictions to numpy if needed
    if torch.is_tensor(predictions):
        pred_probs = predictions.detach().cpu().numpy().squeeze()
    else:
        pred_probs = predictions.squeeze()

    # Use provided timestamps that match the model's frame timing
    pred_length = len(pred_probs)
    pred_times = timestamps

    # Downsample ground truth
    gt_downsampled = downsample_to_prediction_frames(
        ground_truth_mask, pred_length, is_binary=True
    )

    # Downsample audio
    audio_downsampled = downsample_to_prediction_frames(audio, pred_length)
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
    data_dir: Path | str,
    file_id: str,
    predictions: np.ndarray,
    model_metadata: dict[str, Any],
    evaluator: SpeechDetectionEvaluator,
    output_dir: None | Path = None,
    plot_figures: bool = False,
) -> EvaluationMetrics:
    """Analyse a single file with audio, ground truth, and predictions.

    Args:
        data_dir: Directory containing audio and annotation files
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

    audio, _, speech_segments = load_audio_and_annotations(
        file_id,
        data_dir,
    )

    # get model frame parameters
    frame_rate_hz = model_metadata["frame_rate_hz"]
    frame_center_start = model_metadata["frame_center_start"]
    frame_center_step = model_metadata["frame_center_step"]
    actual_num_frames = ((len(audio) - 2 * frame_center_start) // frame_center_step) + 1

    # get only the valid portion of predictions
    signal_predictions = predictions[:actual_num_frames]

    all_timestamps = (
        np.arange(len(signal_predictions)) * (1 / frame_rate_hz)
    ) + model_metadata["frame_hop_sec"]

    # Create ground truth mask
    ground_truth_mask = speaking_map(
        timestamps=all_timestamps,
        annotations=speech_segments,
    )

    # Downsample ground truth to match predictions
    # Compute DER metrics
    evaluation_metrics = evaluator.evaluate(
        predictions=signal_predictions, ground_truth=ground_truth_mask
    )

    if plot_figures and output_dir is not None:
        # Create visualization
        plot_analysis(
            file_id,
            audio,
            ground_truth_mask,
            signal_predictions,
            all_timestamps,
            output_dir,
            evaluation_metrics=evaluation_metrics,
        )

    return evaluation_metrics


def get_ground_truth_and_preds(
    model_metadata: dict[str, float | int],
    data_dir: Path,
    file_id: str,
    predictions,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Terser function to retrieve the ground truth labels for a specific file ID along
    with the model predictions where there is overlap with the audio.

    Args:
        model_metadata: metadata dictionary containing frame parameters
            (frame_rate_hz, frame_center_start, frame_center_step, frame_hop_sec).
        data_dir: Path to the data directory containing audio and annotations.
        file_id: ID of the file to retrieve ground truth for.
        predictions: Model predictions for the audio file.

    Returns:
        Tuple containing ground truth labels and corresponding model predictions.
    """
    audio, _, speech_segments = load_audio_and_annotations(
        file_id,
        data_dir,
    )

    # get model frame parameters
    frame_rate_hz = model_metadata["frame_rate_hz"]
    frame_center_start = model_metadata["frame_center_start"]
    frame_center_step = model_metadata["frame_center_step"]
    actual_num_frames = int(
        ((len(audio) - 2 * frame_center_start) // frame_center_step) + 1
    )

    # get only the valid portion of predictions
    signal_predictions = predictions[:actual_num_frames]

    all_timestamps = (
        np.arange(len(signal_predictions)) * (1 / frame_rate_hz)
    ) + model_metadata["frame_hop_sec"]
    # Create ground truth mask
    ground_truth_mask = speaking_map(
        timestamps=all_timestamps,
        annotations=speech_segments,
    )
    return ground_truth_mask, signal_predictions
