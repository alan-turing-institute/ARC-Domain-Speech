from pathlib import Path
from typing import Any, cast

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dr_sad.data.callhome_utils import load_callhome
from dr_sad.data.sampler import StratifiedSampler
from dr_sad.data.splitting import stratified_splitter
from dr_sad.data.utils import collate_padded

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "callhome"


class DrSadDataset(Dataset):  # type: ignore[misc]
    """
    A dataset class for the DR-SAD project.

    Args:
        Dataset: The base dataset class from PyTorch.
    """

    def __init__(self, data: pd.DataFrame | str, **dataset_gen_kwargs):
        """
        Initialize the DrSadDataset, takes either a DataFrame or a dataset name.

        Args:
            data: The data to use for the dataset, either as a DataFrame or a string
            representing the dataset name.
        """
        if isinstance(data, str):
            data = self.get_data(data, **dataset_gen_kwargs)
        self.data = data
        self.key_list = list(data.index)

    @staticmethod
    def get_data(dataset_name, **dataset_gen_kwargs) -> pd.DataFrame:
        """
        Get the data for a specific dataset.

        Args:
            dataset_name: The name of the dataset to load.

        Raises:
            ValueError: If the dataset name is unknown.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if dataset_name == "callhome":
            return load_callhome(DATA_DIR, **dataset_gen_kwargs)

        err_msg = f"Unknown dataset name: {dataset_name}"
        raise ValueError(err_msg)

    def train_test_split(
        self, val_ratio: float = 0.1, test_ratio: float = 0.1, random_state: int = 42
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            val_ratio: Proportion of data to use for validation. Defaults to 0.1.
            test_ratio: Proportion of data to use for testing. Defaults to 0.1.
            random_state: Random seed for reproducibility. Defaults to 42.

        Returns:
            tuple[DrSadDataset, DrSadDataset, DrSadDataset]: The training, validation,
            and test datasets.
        """
        domains_series = self.data["domains"]
        domains_series.index = domains_series.index.astype(str)

        train_keys, val_keys, test_keys = stratified_splitter(
            domains_series,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_state,
        )

        # Get the indices in the dataset for these keys
        train_indices = [self.key_list.index(int(k)) for k in train_keys]
        val_indices = [self.key_list.index(int(k)) for k in val_keys]
        test_indices = [self.key_list.index(int(k)) for k in test_keys]

        # Create DrSadDataset objects
        return (
            DrSadDataset(self.data.iloc[train_indices]),
            DrSadDataset(self.data.iloc[val_indices]),
            DrSadDataset(self.data.iloc[test_indices]),
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        return cast(dict[str, Any], self.data.iloc[idx].to_dict())


def get_dataloaders(
    dataset: DrSadDataset,
    batch_size: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    **dataloader_kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training, validation, and testing.

    Args:
        dataset (DrSadDataset): The full dataset to split and load.
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 4.
        val_ratio (float, optional): Proportion of data to use for validation.
        Defaults to 0.1.
        test_ratio (float, optional): Proportion of data to use for testing.
        Defaults to 0.1.
        random_state (int, optional): Random seed for reproducibility.
        Defaults to 42.
        **dataloader_kwargs: Additional keyword arguments for DataLoader.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test
        dataloaders.
    """

    train, val, test = dataset.train_test_split(
        val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state
    )

    # get samplers
    train_domains = train.data["domains"].tolist()
    train_sampler = StratifiedSampler(
        domains=train_domains,
        shuffle=True,
    )
    val_domains = val.data["domains"].tolist()
    val_sampler = StratifiedSampler(
        domains=val_domains,
        shuffle=True,
    )
    test_domains = test.data["domains"].tolist()
    test_sampler = StratifiedSampler(
        domains=test_domains,
        shuffle=True,
    )
    # create dataloaders
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_padded,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_padded,
        **dataloader_kwargs,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=collate_padded,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader
