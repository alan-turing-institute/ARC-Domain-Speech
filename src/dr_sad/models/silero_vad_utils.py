import torch
from torch.jit._script import RecursiveScriptModule


def get_probs(
    model: RecursiveScriptModule,
    waveform: torch.Tensor,
    window_size_samples: int,
    sampling_rate: int,
):
    """
    Get speech probabilities from a waveform using a Silero VAD model.

    Args:
        model: The Silero VAD TorchScript model for voice activity detection.
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
            # Pad the last chunk with zeros
            pad_size = window_size_samples - len(chunk)
            chunk = torch.cat(
                [chunk, torch.zeros(pad_size, dtype=chunk.dtype, device=chunk.device)]
            )
        speech_prob: torch.Tensor = model(chunk, sampling_rate)
        speech_probs.append(speech_prob.item())
    model.reset_states()
    return speech_probs


def load_silerovad_model(
    sampling_rate: int,
    cache_dir: None | str = None,
) -> tuple[RecursiveScriptModule, dict[str, int | float]]:
    """
    Load the Silero VAD model using torch.hub.

    Args:
        sampling_rate: sampling rate of the audio data
        cache_dir: Optional path to cache the model

    Returns:
        tuple containing the model, VAD iterator, and metadata dictionary
    """
    if cache_dir is not None:
        torch.hub.set_dir(cache_dir)

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )

    return model, get_silerovad_metadata(sampling_rate)


def get_silerovad_metadata(sample_rate: int) -> dict[str, int | float]:
    """
    Given data sample rate, generate silerovad metadata dict.

    Args:
        sample_rate: int, the sampling rate of the audio data

    Raises:
        ValueError: if ``sample_rate`` is not one of the supported values {16000, 8000}.

    Returns:
        dict containing metadata for SileroVAD model
    """
    if sample_rate == 16000:
        window_size_samples = 512
    elif sample_rate == 8000:
        window_size_samples = 256
    else:
        err_msg = f"Unsupported sample rate for SileroVAD: {sample_rate}"
        raise ValueError(err_msg)

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
