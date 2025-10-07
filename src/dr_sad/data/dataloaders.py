from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from dr_sad.data.callhome_utils import call_home_preprocess
from dr_sad.data.sampler import StratifiedSampler

DOMAIN_LANGUAGES = {"eng": 0, "deu": 1, "spa": 2, "jpn": 3, "zho": 4}


def _callhome_dataloader(languages: list[str] | None = None, **kwargs) -> list[Dataset]:
    """
    Load and preprocess the CallHome dataset for the specified languages.

    Args:
        languages: List of language codes to load (e.g., ["eng", "spa"]).

    Returns:
        Preprocessed dataset ready for training.
    """
    # Load datasets for specified languages
    if languages is None:
        languages = ["eng"]
    datasets = [load_dataset("talkbank/callhome", lang) for lang in languages]

    # Process each dataset separately
    processed_datasets = []
    for dataset, lang in zip(datasets, languages, strict=True):
        # Get the data split
        data = dataset["data"]

        # Add language column using add_column
        data_with_lang = data.add_column(
            "domains", [DOMAIN_LANGUAGES[lang]] * len(data)
        )
        # Skip ClassLabel conversion for now - keep as string to avoid PyArrow issues
        # Preprocess this dataset
        processed_dataset: Dataset = data_with_lang.map(
            call_home_preprocess,
            batch_size=1,
            keep_in_memory=False,
            writer_batch_size=5,
            remove_columns=["timestamps_start", "timestamps_end", "speakers"],
            ).rename_column("audio", "waveforms")

        processed_datasets.append(processed_dataset)

    return processed_datasets


def train_test_split(
    datasets: list[Dataset], val_size: float = 0.1, test_size: float = 0.1, **kwargs
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        dataset: The full dataset to split.
        val_size: Proportion of the dataset to use for validation.
        test_size: Proportion of the dataset to use for testing.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    train_splits = []
    val_splits = []
    test_splits = []

    # Shuffle the dataset
    for dataset in datasets:
        shuffled_dataset = dataset.shuffle(seed=42)
        train_split, non_train_splits = shuffled_dataset.train_test_split(
            test_size=val_size + test_size
        ).values()
        val_split, test_split = non_train_splits.train_test_split(
            test_size=test_size / (val_size + test_size)
        ).values()
        train_splits.append(train_split)
        val_splits.append(val_split)
        test_splits.append(test_split)

    train_split = concatenate_datasets(train_splits)
    val_split = concatenate_datasets(val_splits)
    test_split = concatenate_datasets(test_splits)

    return train_split, val_split, test_split


def get_callhome_dataset(languages: list[str] | None = None, **kwargs) -> Dataset:
    """
    Load and preprocess the CallHome dataset for the specified languages.

    Args:
        languages: List of language codes to load (e.g., ["eng", "spa"]).

    Returns:
        Preprocessed dataset ready for training.
    """

    train, val, test = train_test_split(
        _callhome_dataloader(languages, **kwargs), **kwargs
    )

    return DatasetDict({"train": train, "validation": val, "test": test})


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
        audio_samples = sample[key].get_all_samples()
        audio_data = audio_samples.data  # This is the actual tensor

        # Ensure 2D: [channels, samples]
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)  # Add channel dimension

        audio_tensors.append(audio_data)
        audio_lengths.append(audio_data.shape[-1])  # Last dim is time

    # Pad to same length for batch processing
    max_length = max(audio_lengths)
    padded_audio = []
    audio_masks = []

    for audio, length in zip(audio_tensors, audio_lengths, strict=False):
        # Pad to max length
        if audio.shape[-1] < max_length:
            padding = max_length - audio.shape[-1]
            padded_tensor = torch.nn.functional.pad(audio, (0, padding))
        else:
            padded_tensor = audio

        padded_audio.append(padded_tensor)

        # Create attention mask (True for real audio, False for padding)
        mask = torch.ones(max_length, dtype=torch.bool)
        if length < max_length:
            mask[length:] = False
        audio_masks.append(mask)

    collated_batch[key] = torch.stack(padded_audio)
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
                [sample[key] for sample in batch], dtype=torch.long
            )
        else:
            # nothing else needs special handling - just collate as list
            collated_batch[key] = [sample[key] for sample in batch]

    return collated_batch


def get_callhome_dataloader(dataset: Dataset, **dataloader_kwargs) -> DataLoader:
    """
    Create a DataLoader for the CallHome dataset.

    Args:
        dataset: The dataset to load
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        A DataLoader instance for the dataset.
    """
    # Extract shuffle parameter before creating sampler
    shuffle = dataloader_kwargs.pop("shuffle", True)

    stratified_sampler = StratifiedSampler(
        domains=dataset["domains"],
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        sampler=stratified_sampler,
        collate_fn=collate_padded,
        **dataloader_kwargs,
    )
