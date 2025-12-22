# for type hints
import torch
from silero_vad.utils_vad import VADIterator
from torch.jit._script import RecursiveScriptModule


def get_probs(
    model: RecursiveScriptModule,
    vad_iterator: VADIterator,
    waveform: torch.Tensor,
    window_size_samples: int,
    sampling_rate: int,
):
    """
    Get speech probabilities from a waveform using a Silero VAD model.

    Args:
        model: silerovad model
        vad_iterator: Silero VAD iterator
        waveform: torch tensor of shape (num_samples,) or (1, num_samples)
        window_size_samples: int, size of the window in samples
        sampling_rate: int, sampling rate of the waveform

    Returns:
        List of speech probabilities for each window
    """
    speech_probs = []
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    for i in range(0, len(waveform), window_size_samples):
        chunk = waveform[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob: torch.Tensor = model(chunk, sampling_rate)
        speech_probs.append(speech_prob.item())
    vad_iterator.reset_states()
    return speech_probs


def load_silerovad_model(
    sampling_rate,
) -> tuple[RecursiveScriptModule, VADIterator, dict[str, int | float]]:
    """
    Using torch hub load the silerovad model

    Args:
        sampling_rate: sampling rate of the audio data

    Returns:
        tuple containing the model, VAD iterator, and metadata dictionary
    """
    model, (_, _, _, VADIterator, _) = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad"
    )

    vad_iterator = VADIterator(model, sampling_rate=sampling_rate)

    return model, vad_iterator, get_silerovad_metadata(sampling_rate)


def get_silerovad_metadata(sample_rate: int) -> dict[str, int | float]:
    """
    given data sample rate generate silerovad metadata dict

    Args:
        sample_rate: int, the sampling rate of the audio data

    Returns:
        dict containing metadata for SileroVAD model
    """
    window_size_samples = 512 if sample_rate == 16000 else 256
    window_length = window_size_samples / sample_rate

    # Save model metadata
    return {
        "sample_rate": sample_rate,
        "frame_hop_samples": window_size_samples,
        "frame_hop_sec": window_length,
        "frame_rate_hz": sample_rate / window_size_samples,
        "frame_center_start": window_size_samples // 2,
        "frame_center_step": window_size_samples,
        "receptive_field_samples": window_size_samples,
        "receptive_field_sec": window_size_samples / sample_rate,
    }
