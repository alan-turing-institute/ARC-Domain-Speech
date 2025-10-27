__all__ = (
    "DrSadDataset",
    "domain_split_dataloaders",
    "from_keys_dataloaders",
    "make_dataloader",
    "train_test_split_dataloaders",
)

from typing import Any, cast

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dr_sad.data.sampler import StratifiedSampler
from dr_sad.data.splitting import stratified_splitter
from dr_sad.data.utils import collate_padded


class DrSadDataset(Dataset):  # type: ignore[misc]
    """
    A dataset class for the DR-SAD project.

    Args:
        Dataset: The base dataset class from PyTorch.
    """

    def __init__(self, data: pd.DataFrame, domain: int | None = None):
        """
        Initialize the DrSadDataset.

        Args:
            data (pd.DataFrame): The data to use for the dataset.
            domain (int, optional): The domain index for the dataset.
        """
        self.data = data
        self.domain = domain
        self.key_list = list(data.index)

    @classmethod
    def from_split_domain(
        cls,
        data: pd.DataFrame,
        train_keys: list[str],
        val_keys: list[str],
        test_keys: list[str],
        domain: int,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Create a DrSadDataset for a specific domain.

        Args:
            data (pd.DataFrame): The full dataset to filter.
            train_keys (list): List of keys for the training set.
            val_keys (list): List of keys for the validation set.
            test_keys (list): List of keys for the test set.
            domain (int): The domain index to filter by.

        Returns:
            train (DrSadDataset): Training dataset excluding the specified domain.
            val (DrSadDataset): Validation dataset excluding the specified domain.
            test (DrSadDataset): Test dataset excluding the specified domain.
            domain_data (DrSadDataset): Dataset containing only the specified domain.
        """
        domain_keys = data[data["domains"] == domain].index.to_list()
        train_data = data.loc[list(set(train_keys) - set(domain_keys))]
        val_data = data.loc[list(set(val_keys) - set(domain_keys))]
        test_data = data.loc[list(set(test_keys) - set(domain_keys))]
        domain_data = data.loc[domain_keys]
        return (
            cls(train_data, domain=domain),
            cls(val_data, domain=domain),
            cls(test_data, domain=domain),
            cls(domain_data, domain=domain),
        )

    @classmethod
    def from_splitting_keys(
        cls,
        data: pd.DataFrame,
        train_keys: list[str],
        val_keys: list[str],
        test_keys: list[str],
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Create training, validation, and test datasets from splitting keys.

        Args:
            data (pd.DataFrame): The full dataset to split.
            train_keys (list): List of keys for the training set.
            val_keys (list): List of keys for the validation set.
            test_keys (list): List of keys for the test set.

        Returns:
            train (DrSadDataset): Training dataset.
            val (DrSadDataset): Validation dataset.
            test (DrSadDataset): Test dataset.
        """
        train_data = data.loc[train_keys]
        val_data = data.loc[val_keys]
        test_data = data.loc[test_keys]
        return cls(train_data), cls(val_data), cls(test_data)

    @classmethod
    def from_train_test_split(
        cls,
        data: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int | None = None,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            val_ratio: Proportion of data to use for validation. Defaults to 0.1
            test_ratio: Proportion of data to use for testing. Defaults to 0.2.
            random_seed: Random seed for reproducibility.
                Defaults to None (no seed).

        Returns:
            train (DrSadDataset): Training dataset.
            val (DrSadDataset): Validation dataset.
            test (DrSadDataset): Test dataset.
        """
        if random_seed is None:
            random_seed = np.random.randint(0, 1_000_000)
        domains_series = data["domains"]
        key_list = list(data.index)
        domains_series.index = domains_series.index.astype(str)

        train_keys, val_keys, test_keys = stratified_splitter(
            domains_series,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        # Get the indices in the dataset for these keys
        train_indices = [key_list.index(int(k)) for k in train_keys]
        val_indices = [key_list.index(int(k)) for k in val_keys]
        test_indices = [key_list.index(int(k)) for k in test_keys]

        # Create DrSadDataset objects
        return (
            cls(data.iloc[train_indices]),
            cls(data.iloc[val_indices]),
            cls(data.iloc[test_indices]),
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        return cast(dict[str, Any], self.data.iloc[idx].to_dict())


def make_dataloader(
    dataset: DrSadDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    random_state: np.random.RandomState | None = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader for the given DrSadDataset.

    Args:
        dataset (DrSadDataset): The dataset to load.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        random_state (np.random.RandomState, optional): Random state for shuffling.
            Defaults to None.
        **dataloader_kwargs: Additional keyword arguments to pass to DataLoader.

    Returns:
        DataLoader: The created DataLoader.
    """

    sampler = StratifiedSampler(
        domains=dataset.data["domains"].tolist(),
        shuffle=shuffle,
        generator=random_state,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_padded,
        **dataloader_kwargs,
    )


def train_test_split_dataloaders(
    data: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 4,
    random_seed: int | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing.

    Args:
        data (pd.DataFrame): The full dataset to split and load.
            This must contain "waveforms", "annotations", and "domains" columns.
        val_ratio (float, optional): Proportion of data to use for validation.
            Defaults to 0.1
        test_ratio (float, optional): Proportion of data to use for testing.
            Defaults to 0.2
        batch_size (int, optional): Batch size for the dataloaders.
            Defaults to 4.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to None (no seed).
        dataloader_kwargs (dict, optional): Additional keyword arguments to pass
            to the DataLoader constructor. Defaults to {}.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    train, val, test = DrSadDataset.from_train_test_split(
        data, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)
    val_rng = np.random.default_rng(random_seed + 1)
    test_rng = np.random.default_rng(random_seed + 2)

    # Create dataloaders
    train_loader = make_dataloader(
        train,
        batch_size=batch_size,
        shuffle=True,
        random_state=train_rng,
        **dataloader_kwargs,
    )
    val_loader = make_dataloader(
        val,
        batch_size=batch_size,
        shuffle=False,
        random_state=val_rng,
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        random_state=test_rng,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader


def from_keys_dataloaders(
    data: pd.DataFrame,
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    batch_size: int = 4,
    random_seed: int | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing from keys.

    Args:
        data (pd.DataFrame): The full dataset to split and load.
            This must contain "waveforms", "annotations", and "domains" columns.
        train_keys (list): List of keys for the training set.
        val_keys (list): List of keys for the validation set.
        test_keys (list): List of keys for the test set.
        batch_size (int, optional): Batch size for the dataloaders.
            Defaults to 4.
        random_seed (int, optional): Random seed for reproducibility.
            Defaults to None (no seed).
        dataloader_kwargs (dict, optional): Additional keyword arguments to pass
            to the DataLoader constructor. Defaults to {}.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    train, val, test = DrSadDataset.from_splitting_keys(
        data, train_keys, val_keys, test_keys
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)
    val_rng = np.random.default_rng(random_seed + 1)
    test_rng = np.random.default_rng(random_seed + 2)
    # Create dataloaders
    train_loader = make_dataloader(
        train,
        batch_size=batch_size,
        shuffle=True,
        random_state=train_rng,
        **dataloader_kwargs,
    )
    val_loader = make_dataloader(
        val,
        batch_size=batch_size,
        shuffle=False,
        random_state=val_rng,
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        random_state=test_rng,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader


def domain_split_dataloaders(
    data: pd.DataFrame,
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    domain: int,
    batch_size: int = 4,
    random_seed: int | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, testing, and a specific domain.

    Args:
        data (pd.DataFrame): The full dataset to split and load.
            This must contain "waveforms", "annotations", and "domains" columns.
        train_keys (list): List of keys for the training set.
        val_keys (list): List of keys for the validation set.
        test_keys (list): List of keys for the test set.
        domain (int): The domain index to filter by.
        batch_size (int, optional): Batch size for the dataloaders.
            Defaults to 4.
        random_seed (int, optional): Random seed for reproducibility.
            Defaults to None (no seed).
        dataloader_kwargs (dict, optional): Additional keyword arguments to pass
            to the DataLoader constructor. Defaults to {}.

    Returns:
        train_loader (DataLoader): DataLoader for the training set excluding the domain.
        val_loader (DataLoader): DataLoader for the validation set excluding the domain.
        test_loader (DataLoader): DataLoader for the test set excluding the domain.
        domain_loader (DataLoader): DataLoader for the specified domain.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    train, val, test, domain_data = DrSadDataset.from_split_domain(
        data, train_keys, val_keys, test_keys, domain
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)
    val_rng = np.random.default_rng(random_seed + 1)
    test_rng = np.random.default_rng(random_seed + 2)
    domain_rng = np.random.default_rng(random_seed + 3)

    # Create dataloaders
    train_loader = make_dataloader(
        train,
        batch_size=batch_size,
        shuffle=True,
        random_state=train_rng,
        **dataloader_kwargs,
    )
    val_loader = make_dataloader(
        val,
        batch_size=batch_size,
        shuffle=False,
        random_state=val_rng,
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        random_state=test_rng,
        **dataloader_kwargs,
    )
    domain_loader = make_dataloader(
        domain_data,
        batch_size=batch_size,
        shuffle=False,
        random_state=domain_rng,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader, domain_loader
