import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    DrSadDataset,
    make_dataloader,
    train_test_split_dataloaders,
)
from dr_sad.data.sampler import StratifiedSampler


class TestDrSadDataset:
    def test_simple_dataset_length(self):
        """Test that the length of the dataset matches the input DataFrame."""
        # Create a test DataFrame
        test_df = pd.DataFrame(
            {
                "waveforms": [
                    0.3 * np.ones(200),
                    0.4 * np.ones(200),
                    0.5 * np.ones(200),
                ],
                "annotations": [[(0.1, 1.1)], [(0.2, 1.2)], [(0.3, 1.3)]],
                "domains": [0, 1, 0],
            }
        )

        dataset = DrSadDataset(test_df)
        assert len(dataset) == len(test_df)

    def test_simple_dataset_getitem(self):
        """Test that __getitem__ returns the correct data."""
        # Create a test DataFrame
        test_df = pd.DataFrame(
            {
                "waveforms": [
                    0.3 * np.ones(200),
                    0.4 * np.ones(200),
                    0.5 * np.ones(200),
                ],
                "annotations": [[(0.1, 1.1)], [(0.2, 1.2)], [(0.3, 1.3)]],
                "domains": [0, 1, 0],
            }
        )

        dataset = DrSadDataset(test_df)

        for item in dataset:
            assert "waveforms" in item
            assert isinstance(item["waveforms"], np.ndarray)
            assert "annotations" in item
            assert isinstance(item["annotations"], list)
            assert "domains" in item
            assert isinstance(item["domains"], int)

    def test_load_example_dataset(self, test_dataset):
        """Test loading the test dataset using DrSadDataset."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        dataset = DrSadDataset(data)

        # Check basic structure
        assert len(dataset) == 20

        # Check first item
        first_item = dataset[0]
        assert "waveforms" in first_item
        assert "annotations" in first_item
        assert "domains" in first_item

        # Check that annotations are correct for the first file
        first_annotations = first_item["annotations"]
        assert len(first_annotations) == 2
        assert first_annotations[0] == (0.5, 1.0)

    def test_dataset_split(self, test_dataset):
        """Test that DrSadDataset.train_test_split correctly splits the dataset."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train, val, test = DrSadDataset.train_test_split(
            data, val_ratio=0.2, test_ratio=0.2, random_seed=123
        )

        # Check that splits are non-overlapping
        train_indices = set(train.data.index)
        val_indices = set(val.data.index)
        test_indices = set(test.data.index)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)


class TestDataloader:
    def test_make_dataloader(self, test_dataset):
        """Test that make_dataloader creates a properly configured DataLoader."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        dataset = DrSadDataset(data)

        # Test basic dataloader creation
        dataloader = make_dataloader(dataset, batch_size=2, shuffle=True)

        # Check that it returns a DataLoader instance
        assert isinstance(dataloader, DataLoader)

        # Check batch size
        assert dataloader.batch_size == 2

        # Check that sampler is StratifiedSampler
        assert isinstance(dataloader.sampler, StratifiedSampler)

        # Test that we can iterate through batches
        batches = list(dataloader)
        assert len(batches) > 0

        # Check structure of first batch
        first_batch = batches[0]
        assert "waveforms" in first_batch
        assert "annotations" in first_batch
        assert "domains" in first_batch

    def test_make_dataloader_with_custom_params(self, test_dataset):
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        dataset = DrSadDataset(data)
        # Test with custom parameters
        random_state = np.random.RandomState(42)
        dataloader_custom = make_dataloader(
            dataset,
            batch_size=3,
            shuffle=False,
            random_state=random_state,
            num_workers=0,
        )

        assert dataloader_custom.batch_size == 3
        assert dataloader_custom.num_workers == 0


class TestTrainTestSplitDataloaders:
    def test_train_test_split_dataloaders_basic(self, test_dataset):
        """Test basic functionality of train_test_split_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data, batch_size=2, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        # Check that all returned objects are DataLoaders
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Check batch sizes
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2

        # Check that all have StratifiedSampler
        assert isinstance(train_loader.sampler, StratifiedSampler)
        assert isinstance(val_loader.sampler, StratifiedSampler)
        assert isinstance(test_loader.sampler, StratifiedSampler)

    def test_train_test_split_dataloaders_custom_params(self, test_dataset):
        """Test train_test_split_dataloaders with custom parameters."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        custom_kwargs = {"num_workers": 0}

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=3,
            val_ratio=0.15,
            test_ratio=0.25,
            random_seed=123,
            dataloader_kwargs=custom_kwargs,
        )

        # Check custom batch size
        assert train_loader.batch_size == 3
        assert val_loader.batch_size == 3
        assert test_loader.batch_size == 3

        # Check custom dataloader kwargs were applied
        assert train_loader.num_workers == 0
        assert val_loader.num_workers == 0
        assert test_loader.num_workers == 0

    def test_train_test_split_dataloaders_iteration(self, test_dataset):
        """Test that we can iterate through the created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data, batch_size=2, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch]:
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2
