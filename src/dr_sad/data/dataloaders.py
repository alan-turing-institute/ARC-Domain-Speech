__all__ = (
    "DrSadDataset",
    "domain_split_dataloaders",
    "from_keys_dataloaders",
    "make_dataloader",
    "single_domain_dataloaders",
    "train_test_split_dataloaders",
)

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dr_sad.data.noise import NoiseBuilder, generate_noise_kwargs_list
from dr_sad.data.sampler import StratifiedSampler
from dr_sad.data.splitting import stratified_splitter
from dr_sad.data.utils import collate_padded


class DrSadDataset(Dataset):  # type: ignore[misc]
    """
    A dataset class for the DR-SAD project.

    Args:
        Dataset: The base dataset class from PyTorch.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        domain: int | None = None,
        time_slice: float | None = None,
        sample_rate: int = 16_000,
        noise_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the DrSadDataset.

        This also supports cutting waveforms to a specific length in seconds.
        When this is done the new file ids are created by appending
        "-XX" to the original file id, where XX is a zero-padded index in hexadecimal.
        The waveforms that are shorter than the specified length are kept as is.
        The waveforms that are longer than the specified length split into multiple
        segments of the specified length positioning them in the mid point.

        Noise augmentation can be added by providing noise_kwargs.

        Args:
            data (pd.DataFrame): The data to use for the dataset.
            domain (int, optional): The domain index for the dataset.
            time_slice (float, optional): Cut the waveforms to this length in seconds.
            sample_rate (int, optional): The sample rate of the waveforms.
                Defaults to 16_000 Hz.
            noise_kwargs: (dict, optional): Keyword arguments for adding noise.
                Defaults to None, which means no noise is added.
                - noise_dir (Path | str): The directory containing noise audio files.
                - snr_db (float, tuple[float, float]): The desired signal-to-noise ratio
                    in decibels (dB).
                - simultaneous (int, optional): The number of noise files to use.
                - start_choice (bool, optional): Randomise starting point for cropping.
                - noise_files_list (list[str], optional): Specific noise file names.
        """
        self.noise_kwargs = noise_kwargs
        if noise_kwargs is not None:
            noise_builder = NoiseBuilder(**noise_kwargs)
            augmented_waveforms = pd.Series(dtype=object)
            for key, waveform in data["waveforms"].items():
                augmented_waveforms.loc[key] = noise_builder.add_noise(waveform)
            # Replace the original waveforms with the augmented ones
            data = data.copy()
            data["waveforms"] = augmented_waveforms

        if time_slice is not None:
            if time_slice <= 0:
                msg = "time_slice must be a positive value in seconds."
                raise ValueError(msg)

            time_count = round(time_slice * sample_rate)
            self.data = pd.DataFrame(columns=data.columns)
            for row in data.itertuples(index=True, name="Row"):
                file_id = row.Index
                if len(row.waveforms) <= time_count:
                    self.data.loc[f"{file_id}-00"] = {
                        "waveforms": row.waveforms,
                        "annotations": row.annotations,
                        "domains": row.domains,
                    }
                else:
                    num_slices = len(row.waveforms) // time_count
                    start_point = (len(row.waveforms) - num_slices * time_count) // 2
                    for slice_idx in range(num_slices):
                        start_idx = start_point + slice_idx * time_count
                        end_idx = start_idx + time_count
                        sliced_waveform = row.waveforms[start_idx:end_idx]
                        new_file_id = f"{file_id}-{slice_idx:02x}"
                        start_time = start_idx / sample_rate
                        end_time = end_idx / sample_rate
                        new_annotations = []
                        for ann in row.annotations:
                            if ann[0] >= end_time:
                                continue
                            if ann[1] <= start_time:
                                continue
                            new_start = max(0.0, ann[0] - start_time)
                            new_end = min(time_slice, ann[1] - start_time)
                            new_annotations.append((new_start, new_end))
                        self.data.loc[new_file_id] = {
                            "file_id": new_file_id,
                            "waveforms": sliced_waveform,
                            "annotations": new_annotations,
                            "domains": row.domains,
                        }

        else:
            self.data = data
        self.sample_rate = int(sample_rate)
        self.domain = domain
        self.time_slice = time_slice
        self.key_list = list(data.index)

    @classmethod
    def from_target_domain(
        cls,
        data: pd.DataFrame,
        train_keys: list[str],
        val_keys: list[str],
        test_keys: list[str],
        domain: int,
        time_slice: float | None = None,
        sample_rate: int = 16_000,
        noise_kwargs: dict[str, Any] | None = None,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Create a DrSadDataset for the target domain.

        Args:
            data (pd.DataFrame): The full dataset to filter.
            train_keys (list[str]): Indices/keys defining the training subset.
            val_keys (list[str]): Indices/keys defining the validation subset.
            test_keys (list[str]): Indices/keys defining the test subset.
            domain (int): The domain index to filter by.
            time_slice (float, optional): Cut the waveforms to this length in seconds.
            sample_rate (int, optional): The sample rate of the waveforms.
                Defaults to 16_000 Hz.
            noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
                addition. Defaults to None.
        Returns:
            train (DrSadDataset): Training dataset of the specified domain.
            val (DrSadDataset): Validation dataset of the specified domain.
            test (DrSadDataset): Test dataset of the specified domain.
        """

        domain_keys = data[data["domains"] == domain].index.to_list()

        if len(domain_keys) == 0:
            available_domains = sorted(data["domains"].unique().tolist())
            msg = (
                f"Domain {domain} has no associated data. "
                f"Available domains: {available_domains}"
            )
            raise ValueError(msg)

        train_data = data.loc[list(set(train_keys) & set(domain_keys))]
        val_data = data.loc[list(set(val_keys) & set(domain_keys))]
        test_data = data.loc[list(set(test_keys) & set(domain_keys))]

        train_only = noise_kwargs is not None and noise_kwargs.get("train_only", False)
        if train_only:
            train_noise_kwargs: dict[str, Any] | None = {
                k: v
                for k, v in noise_kwargs.items()  # type: ignore[union-attr]
                if k != "train_only"
            }
            val_noise_kwargs: dict[str, Any] | None = None
            test_noise_kwargs: dict[str, Any] | None = None
        else:
            # remove "train_only" key if present and false
            noise_kwargs: dict[str, Any] | None = {
                k: v
                for k, v in noise_kwargs.items()  # type: ignore[union-attr]
                if k != "train_only"
            }
            train_noise_kwargs, val_noise_kwargs, test_noise_kwargs = (
                generate_noise_kwargs_list(noise_kwargs, 3)
            )

        return (
            cls(
                train_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=train_noise_kwargs,
            ),
            cls(
                val_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=val_noise_kwargs,
            ),
            cls(
                test_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=test_noise_kwargs,
            ),
        )

    @classmethod
    def from_split_domain(
        cls,
        data: pd.DataFrame,
        train_keys: list[str],
        val_keys: list[str],
        test_keys: list[str],
        domain: int,
        time_slice: float | None = None,
        sample_rate: int = 16_000,
        noise_kwargs: dict[str, Any] | None = None,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Create a DrSadDataset for a specific domain.

        Args:
            data (pd.DataFrame): The full dataset to filter.
            train_keys (list): List of keys for the training set.
            val_keys (list): List of keys for the validation set.
            test_keys (list): List of keys for the test set.
            domain (int): The domain index to filter by.
            time_slice (float, optional): Cut the waveforms to this length in seconds.
            sample_rate (int, optional): The sample rate of the waveforms.
                Defaults to 16_000 Hz.
            noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
                addition. Defaults to None.

        Returns:
            train (DrSadDataset): Training dataset excluding the specified domain.
            val (DrSadDataset): Validation dataset excluding the specified domain.
            test (DrSadDataset): Test dataset excluding the specified domain.
            domain_data (DrSadDataset): Dataset containing only the specified domain.
        """
        domain_keys = data[data["domains"] == domain].index.to_list()
        if len(domain_keys) == 0:
            available_domains = sorted(data["domains"].unique().tolist())
            msg = (
                f"Domain {domain} has no associated data. "
                f"Available domains: {available_domains}"
            )
            raise ValueError(msg)
        train_data = data.loc[list(set(train_keys) - set(domain_keys))]
        val_data = data.loc[list(set(val_keys) - set(domain_keys))]
        test_data = data.loc[list(set(test_keys) - set(domain_keys))]
        domain_data = data.loc[domain_keys]

        train_only = noise_kwargs is not None and noise_kwargs.get("train_only", False)
        if train_only:
            train_noise_kwargs: dict[str, Any] | None = {
                k: v
                for k, v in noise_kwargs.items()  # type: ignore[union-attr]
                if k != "train_only"
            }
            val_noise_kwargs: dict[str, Any] | None = None
            test_noise_kwargs: dict[str, Any] | None = None
            domain_noise_kwargs: dict[str, Any] | None = None
        else:
            (
                train_noise_kwargs,
                val_noise_kwargs,
                test_noise_kwargs,
                domain_noise_kwargs,
            ) = generate_noise_kwargs_list(noise_kwargs, 4)

        return (
            cls(
                train_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=train_noise_kwargs,
            ),
            cls(
                val_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=val_noise_kwargs,
            ),
            cls(
                test_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=test_noise_kwargs,
            ),
            cls(
                domain_data,
                domain=domain,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=domain_noise_kwargs,
            ),
        )

    @classmethod
    def from_splitting_keys(
        cls,
        data: pd.DataFrame,
        train_keys: list[str],
        val_keys: list[str],
        test_keys: list[str],
        time_slice: float | None = None,
        sample_rate: int = 16_000,
        noise_kwargs: dict[str, Any] | None = None,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Create training, validation, and test datasets from splitting keys.

        Args:
            data (pd.DataFrame): The full dataset to split.
            train_keys (list): List of keys for the training set.
            val_keys (list): List of keys for the validation set.
            test_keys (list): List of keys for the test set.
            time_slice (float, optional): Cut the waveforms to this length in seconds.
            sample_rate (int, optional): The sample rate of the waveforms.
                Defaults to 16_000 Hz.
            noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
                addition. Defaults to None.

        Returns:
            train (DrSadDataset): Training dataset.
            val (DrSadDataset): Validation dataset.
            test (DrSadDataset): Test dataset.
        """
        train_data = data.loc[train_keys]
        val_data = data.loc[val_keys]
        test_data = data.loc[test_keys]

        train_only = noise_kwargs is not None and noise_kwargs.get("train_only", False)
        if train_only:
            train_noise_kwargs: dict[str, Any] | None = {
                k: v
                for k, v in noise_kwargs.items()  # type: ignore[union-attr]
                if k != "train_only"
            }
            val_noise_kwargs: dict[str, Any] | None = None
            test_noise_kwargs: dict[str, Any] | None = None
        else:
            train_noise_kwargs, val_noise_kwargs, test_noise_kwargs = (
                generate_noise_kwargs_list(noise_kwargs, 3)
            )

        return (
            cls(
                train_data,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=train_noise_kwargs,
            ),
            cls(
                val_data,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=val_noise_kwargs,
            ),
            cls(
                test_data,
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=test_noise_kwargs,
            ),
        )

    @classmethod
    def from_train_test_split(
        cls,
        data: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int | None = None,
        time_slice: float | None = None,
        sample_rate: int = 16_000,
        noise_kwargs: dict[str, Any] | None = None,
    ) -> tuple["DrSadDataset", "DrSadDataset", "DrSadDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            val_ratio: Proportion of data to use for validation. Defaults to 0.1
            test_ratio: Proportion of data to use for testing. Defaults to 0.2.
            random_seed: Random seed for reproducibility.
                Defaults to None (no seed).
            time_slice (float, optional): Cut the waveforms to this length in seconds.
            sample_rate (int, optional): The sample rate of the waveforms.
                Defaults to 16_000 Hz.
            noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
                addition. Defaults to None.

        Returns:
            train (DrSadDataset): Training dataset.
            val (DrSadDataset): Validation dataset.
            test (DrSadDataset): Test dataset.
        """
        if random_seed is None:
            random_seed = np.random.randint(0, 1_000_000)
        domains_series = data["domains"]
        domains_series.index = domains_series.index.astype(str)

        train_keys, val_keys, test_keys = stratified_splitter(
            domains_series,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        train_only = noise_kwargs is not None and noise_kwargs.get("train_only", False)
        if train_only:
            train_noise_kwargs: dict[str, Any] | None = {
                k: v
                for k, v in noise_kwargs.items()  # type: ignore[union-attr]
                if k != "train_only"
            }
            val_noise_kwargs: dict[str, Any] | None = None
            test_noise_kwargs: dict[str, Any] | None = None
        else:
            train_noise_kwargs, val_noise_kwargs, test_noise_kwargs = (
                generate_noise_kwargs_list(noise_kwargs, 3)
            )

        # Create DrSadDataset objects
        return (
            cls(
                data.loc[train_keys],
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=train_noise_kwargs,
            ),
            cls(
                data.loc[val_keys],
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=val_noise_kwargs,
            ),
            cls(
                data.loc[test_keys],
                time_slice=time_slice,
                sample_rate=sample_rate,
                noise_kwargs=test_noise_kwargs,
            ),
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        file_id = self.data.index[idx]
        row: dict[str, Any] = self.data.iloc[idx].to_dict()
        return row | {"file_id": str(file_id)}


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
    time_slice: float | None = None,
    sample_rate: int = 16_000,
    noise_kwargs: dict[str, Any] | None = None,
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
        time_slice (float, optional): Cut the waveforms to this length in seconds.
        sample_rate (int, optional): The sample rate of the waveforms.
            Defaults to 16_000 Hz.
        noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
            addition. Defaults to None.
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
        data,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        time_slice=time_slice,
        sample_rate=sample_rate,
        noise_kwargs=noise_kwargs,
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)

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
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
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
    time_slice: float | None = None,
    sample_rate: int = 16_000,
    noise_kwargs: dict[str, Any] | None = None,
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
        time_slice (float, optional): Cut the waveforms to this length in seconds.
        sample_rate (int, optional): The sample rate of the waveforms.
            Defaults to 16_000 Hz.
        noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
            addition. Defaults to None.
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
        data,
        train_keys,
        val_keys,
        test_keys,
        time_slice=time_slice,
        sample_rate=sample_rate,
        noise_kwargs=noise_kwargs,
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)

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
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader


def single_domain_dataloaders(
    data: pd.DataFrame,
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
    domain: int,
    batch_size: int = 4,
    random_seed: int | None = None,
    time_slice: float | None = None,
    sample_rate: int = 16_000,
    noise_kwargs: dict[str, Any] | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
        time_slice (float, optional): Cut the waveforms to this length in seconds.
        sample_rate (int, optional): The sample rate of the waveforms.
            Defaults to 16_000 Hz.
        noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
            addition. Defaults to None.
        dataloader_kwargs (dict, optional): Additional keyword arguments to pass
            to the DataLoader constructor. Defaults to {}.

    Returns:
        train_loader (DataLoader): DataLoader for training from target domain.
        val_loader (DataLoader): DataLoader for validation from target domain.
        test_loader (DataLoader): DataLoader for testing from target domain.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    train, val, test = DrSadDataset.from_target_domain(
        data,
        train_keys,
        val_keys,
        test_keys,
        domain,
        time_slice=time_slice,
        sample_rate=sample_rate,
        noise_kwargs=noise_kwargs,
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)

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
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
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
    time_slice: float | None = None,
    sample_rate: int = 16_000,
    noise_kwargs: dict[str, Any] | None = None,
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
        time_slice (float, optional): Cut the waveforms to this length in seconds.
        sample_rate (int, optional): The sample rate of the waveforms.
            Defaults to 16_000 Hz.
        noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
            addition. Defaults to None.
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
        data,
        train_keys,
        val_keys,
        test_keys,
        domain,
        time_slice=time_slice,
        sample_rate=sample_rate,
        noise_kwargs=noise_kwargs,
    )
    if random_seed is None:
        random_seed = np.random.randint(0, 1_000_000)
    train_rng = np.random.default_rng(random_seed)

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
        **dataloader_kwargs,
    )
    test_loader = make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
    domain_loader = make_dataloader(
        domain_data,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader, domain_loader


def one_test_dataloader(
    data: pd.DataFrame,
    data_keys: list[str],
    domain: int | None = None,
    batch_size: int = 4,
    time_slice: float | None = None,
    sample_rate: int = 16_000,
    noise_kwargs: dict[str, Any] | None = None,
    dataloader_kwargs: dict[str, Any] | None = None,
) -> DataLoader:
    """Create a dataloader for testing only.

    Args:
        data (pd.DataFrame): The full dataset to load.
            This must contain "waveforms", "annotations", and "domains" columns.
        data_keys (list): List of keys for the test set.
        domain (int, optional): The domain index to add as metadata to the dataset.
            This DOES NOT filter the data. Defaults to None.
        batch_size (int, optional): Batch size for the dataloader.
            Defaults to 4.
        time_slice (float, optional): Cut the waveforms to this length in seconds.
        sample_rate (int, optional): The sample rate of the waveforms.
            Defaults to 16_000 Hz.
        noise_kwargs (dict[str, Any], optional): Keyword arguments for noise
            addition. Defaults to None.
        dataloader_kwargs (dict, optional): Additional keyword arguments to pass
            to the DataLoader constructor. Defaults to {}.

    Returns:
        test_loader (DataLoader): DataLoader for the test set.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    test_data = data.loc[data_keys]
    test = DrSadDataset(
        test_data,
        domain=domain,
        time_slice=time_slice,
        sample_rate=sample_rate,
        noise_kwargs=noise_kwargs,
    )

    # Create dataloader
    return make_dataloader(
        test,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
