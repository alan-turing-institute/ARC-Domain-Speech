from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch


def _load_audio_and_annotations(file_id, data_dir="data/callhome"):
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


def _create_ground_truth_mask(audio_length, sample_rate, speech_segments):
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
    signal, sample_rate, prediction_length, frame_rate_hz=160
):
    """Downsample audio to match prediction frame rate."""
    # Calculate the frame hop in samples (approximately)
    frame_hop_samples = sample_rate // frame_rate_hz

    # Downsample by taking averages over frame windows
    downsampled = []
    for i in range(prediction_length):
        start_idx = i * frame_hop_samples
        end_idx = min((i + 1) * frame_hop_samples, len(signal))
        if start_idx < len(signal):
            if end_idx > start_idx:
                frame_avg = np.mean(signal[start_idx:end_idx])
            else:
                frame_avg = 0.0
            downsampled.append(frame_avg)
        else:
            downsampled.append(0.0)

    # Ensure output length matches prediction_length exactly
    downsampled = np.array(downsampled)
    if len(downsampled) < prediction_length:
        # Pad with zeros if too short
        downsampled = np.pad(
            downsampled, (0, prediction_length - len(downsampled)), mode="constant"
        )
    elif len(downsampled) > prediction_length:
        # Trim if too long
        downsampled = downsampled[:prediction_length]
    return downsampled


def plot_analysis(
    file_id,
    audio,
    sample_rate,
    ground_truth_mask,
    predictions,
    output_dir=None,
):
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
        pred_probs = predictions.squeeze().numpy()
    else:
        pred_probs = predictions.squeeze()

    # Downsample ground truth to match prediction frames
    pred_length = len(pred_probs)
    gt_downsampled = _downsample_to_prediction_frames(
        ground_truth_mask, sample_rate, pred_length
    )

    # Upsample predictions and ground truth to audio sample rate for aligned plotting
    audio_len = len(audio)
    # Create time axis for original audio
    time_audio = np.arange(audio_len) / sample_rate

    # For upsampling, use np.interp to match audio length
    pred_time = np.linspace(0, audio_len / sample_rate, pred_length, endpoint=False)
    pred_probs_upsampled = np.interp(time_audio, pred_time, pred_probs)
    gt_time = np.linspace(0, audio_len / sample_rate, pred_length, endpoint=False)
    gt_upsampled = np.interp(time_audio, gt_time, gt_downsampled)

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # Plot audio waveform, scaled and shifted to be centered at 0.5
    audio_normalized = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    # Scale to 0.3 amplitude and shift to center at 0.5
    audio_scaled = (audio_normalized * 0.4) + 0.5
    ax.plot(
        time_audio,
        audio_scaled,
        alpha=0.7,
        color="lightgray",
        linewidth=0.5,
        label="Audio",
    )

    # Plot upsampled ground truth speech activity
    ax.fill_between(
        time_audio,
        np.zeros_like(gt_upsampled),
        gt_upsampled,
        color="green",
        linewidth=2,
        label="Ground Truth",
        alpha=0.5,
    )

    # Plot upsampled model predictions
    ax.plot(
        time_audio,
        pred_probs_upsampled,
        color="red",
        linewidth=2,
        label="Model Predictions",
        alpha=0.8,
    )

    # Formatting
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Speech Activity Detection: {file_id}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Determine output directory and create if needed
    output_dir = Path(".temp") if output_dir is None else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    output_path = output_dir / f"{file_id}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    # Close the figure to free memory and avoid resource leaks
    plt.close(fig)

    return fig


def analyse_file(file_id, predictions, output_dir=None):
    """Analyze a single file with audio, ground truth, and predictions.

    Args:
        file_id: ID of the file to analyze
        predictions: Model predictions for this file
        output_dir: Directory to save plots (if None, uses .temp/)
    """

    audio, sample_rate, _speech_segments = _load_audio_and_annotations(file_id)

    # Create ground truth mask
    ground_truth_mask = _create_ground_truth_mask(
        len(audio), sample_rate, _speech_segments
    )

    # Create visualization
    plot_analysis(
        file_id,
        audio,
        sample_rate,
        ground_truth_mask,
        predictions,
        output_dir,
    )
