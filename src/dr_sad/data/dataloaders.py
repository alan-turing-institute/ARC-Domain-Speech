from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset

from dr_sad.data.callhome_utils import call_home_preprocess

DOMAIN_LANGUAGES = {
    "eng": 0,
    "deu": 1,
    "spa": 2,
    "jpn": 3,
    "zho": 4
}

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
        data_with_lang = data.add_column("language", [DOMAIN_LANGUAGES[lang]]*len(data))

        # Skip ClassLabel conversion for now - keep as string to avoid PyArrow issues
        # Preprocess this dataset
        processed_dataset = data_with_lang.map(
            call_home_preprocess,
            batch_size=1,
            keep_in_memory=False,
            writer_batch_size=5,
            remove_columns=["timestamps_start", "timestamps_end", "speakers"],
        )

        processed_datasets.append(processed_dataset)

    return processed_datasets


def train_test_split(datasets: list[Dataset], val_size: float = 0.1, test_size: float = 0.1, **kwargs) -> tuple[Dataset, Dataset, Dataset]:
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
        dataset = dataset.shuffle(seed=42)
        train_split, non_train_splits = dataset.train_test_split(
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

    return train_test_split(_callhome_dataloader(languages,**kwargs), **kwargs)