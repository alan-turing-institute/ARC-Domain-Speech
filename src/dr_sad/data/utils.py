__all__ = ("audio_collation", "collate_padded", "generate_rttm")

import logging
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def audio_collation(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate audio data from a batch of samples.

    Args:
        batch: List of samples, each containing audio data.

    Returns:
        A dictionary containing collated audio tensors and masks.
    """
    collated_batch = {}
    key = "waveforms"
    # Process AudioDecoder objects into tensors
    audio_tensors = []
    audio_lengths = []

    for sample in batch:
        # Extract audio from AudioDecoder using correct method
        audio_data = torch.tensor(
            sample[key], dtype=torch.float32
        )  # This is the actual tensor
        audio_tensors.append(audio_data)
        audio_lengths.append(audio_data.shape[-1])  # Last dim is time

    # Pad to same length for batch processing
    max_length = max(audio_lengths)
    padded_audio = []

    for audio in audio_tensors:
        # Pad to max length
        if audio.shape[-1] < max_length:
            padding = max_length - audio.shape[-1]
            padded_tensor = torch.nn.functional.pad(audio, (0, padding))
        else:
            padded_tensor = audio

        padded_audio.append(padded_tensor)

    collated_batch[key] = torch.vstack(padded_audio).unsqueeze(
        1
    )  # Add channel dimension
    return collated_batch


# Define a custom collate function to handle variable-sized data
def collate_padded(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function with padding for batch processing.

    Args:
        batch: List of samples, each a dictionary with keys like 'audio', 'segments',
               'labels', and 'language'.

    Returns:
        A dictionary containing collated and padded tensors for each key.
    """

    collated_batch = {}

    # Handle each field in the batch
    for key in batch[0]:
        if key == "waveforms":
            # these need to be batch processed
            audio_collated = audio_collation(batch)
            collated_batch.update(audio_collated)

        elif key == "domains":
            # so do these for domain classification
            collated_batch[key] = torch.tensor(
                [sample[key] for sample in batch], dtype=torch.float32
            )
        else:
            # nothing else needs special handling - just collate as list
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch


def generate_rttm(
    data: dict[str, Any],
    rttm_dir: Path,
    file_id: str,
    channel_id: int = 1,
) -> None:
    beginnings = data["timestamps_start"]
    ends = data["timestamps_end"]
    speakers = data["speakers"]
    filtered = []
    has_negative_length = False
    for idx, (start, end, speaker) in enumerate(
        zip(beginnings, ends, speakers, strict=True)
    ):
        length = end - start
        if length < 0:
            logging_msg = f"file_id={file_id}, index={idx}, start={start}, end={end}\n"
            logger.warning(logging_msg)
            has_negative_length = True
        else:
            filtered.append((start, length, speaker))
    if has_negative_length:
        log_msg = f"Skipped RTTM file for {file_id} due to problematic segments."
        logger.warning(log_msg)
        return False
    with open(rttm_dir / f"{file_id}.rttm", "w") as f:
        for begin, length, speaker in filtered:
            if speaker != "None":
                f.write(
                    f"SPEAKER {file_id} {channel_id} {begin:.3f} {length:.3f} <NA> "
                    f"<NA> {speaker} <NA> <NA>\n"
                )
    return True
