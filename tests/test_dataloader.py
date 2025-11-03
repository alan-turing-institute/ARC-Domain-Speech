import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    DrSadDataset,
    domain_split_dataloaders,
    from_keys_dataloaders,
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
            assert "file_id" in item
            assert isinstance(item["file_id"], str)

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
        assert "file_id" in first_item

        assert first_item["file_id"] == "TEST_0001"

        # Check that annotations are correct for the first file
        first_annotations = first_item["annotations"]
        assert len(first_annotations) == 2
        assert first_annotations[0] == (0.5, 1.0)

    def test_from_dataset_split(self, test_dataset):
        """Test that DrSadDataset.train_test_split correctly splits the dataset."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train, val, test = DrSadDataset.from_train_test_split(
            data, val_ratio=0.2, test_ratio=0.2, random_seed=123
        )

        # Check that splits are non-overlapping
        train_indices = set(train.data.index)
        val_indices = set(val.data.index)
        test_indices = set(test.data.index)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

    def test_from_splitting_keys(self, test_dataset):
        """Test that DrSadDataset.from_splitting_keys correctly creates datasets."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        train, val, test = DrSadDataset.from_splitting_keys(
            data, train_keys, val_keys, test_keys
        )

        # Check that splits are non-overlapping
        train_indices = set(train.data.index)
        val_indices = set(val.data.index)
        test_indices = set(test.data.index)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

    def test_from_split_domain(self, test_dataset):
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        domain = 1

        train, val, test, domain_data = DrSadDataset.from_split_domain(
            data, train_keys, val_keys, test_keys, domain
        )

        # domain_data should contain only samples from the chosen domain
        domain_indices = set(domain_data.data.index)
        assert len(domain_indices) > 0
        for idx in domain_indices:
            assert int(data.loc[idx, "domains"]) == domain

        # returned datasets should carry the domain attribute
        assert train.domain == domain
        assert val.domain == domain
        assert test.domain == domain
        assert domain_data.domain == domain

        # domain keys must be excluded from train/val/test
        assert domain_indices.isdisjoint(set(train.data.index))
        assert domain_indices.isdisjoint(set(val.data.index))
        assert domain_indices.isdisjoint(set(test.data.index))

        # train/val/test should equal provided keys minus domain keys
        expected_train = set(train_keys) - domain_indices
        expected_val = set(val_keys) - domain_indices
        expected_test = set(test_keys) - domain_indices

        assert set(train.data.index) == expected_train
        assert set(val.data.index) == expected_val
        assert set(test.data.index) == expected_test

        # union of all returned indices should equal the original dataset indices
        all_returned = (
            set(train.data.index)
            | set(val.data.index)
            | set(test.data.index)
            | domain_indices
        )
        assert all_returned == set(data.index)

    def test_from_split_domain_raises_error_for_invalid_domain(self, test_dataset):
        """Test that from_split_domain raises ValueError for domain with no data."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        # Use a domain that doesn't exist in the data
        invalid_domain = 99

        # Should raise ValueError
        with pytest.raises(ValueError, match=r"Domain 99 has no associated data"):
            DrSadDataset.from_split_domain(
                data, train_keys, val_keys, test_keys, invalid_domain
            )


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


class TestFromKeysDataloaders:
    def test_from_keys_dataloaders_basic(self, test_dataset):
        """Test basic functionality of from_keys_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            batch_size=2,
            random_seed=42,
        )

        # Check that all returned objects are DataLoaders
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Check batch sizes
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2

    def test_from_keys_dataloaders_iteration(self, test_dataset):
        """Test that we can iterate through the created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            batch_size=2,
            random_seed=42,
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


class TestDomainSplitDataloaders:
    def test_domain_split_dataloaders_basic(self, test_dataset):
        """Test basic functionality of domain_split_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1

        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=2,
            random_seed=42,
        )

        # Check that all returned objects are DataLoaders
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(domain_loader, DataLoader)

        # Check batch sizes
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2
        assert domain_loader.batch_size == 2

        # Check that all have StratifiedSampler
        assert isinstance(train_loader.sampler, StratifiedSampler)
        assert isinstance(val_loader.sampler, StratifiedSampler)
        assert isinstance(test_loader.sampler, StratifiedSampler)
        assert isinstance(domain_loader.sampler, StratifiedSampler)

    def test_domain_split_dataloaders_custom_params(self, test_dataset):
        """Test domain_split_dataloaders with custom parameters."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1
        custom_kwargs = {"num_workers": 0}

        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=3,
            random_seed=123,
            dataloader_kwargs=custom_kwargs,
        )

        # Check custom batch size
        assert train_loader.batch_size == 3
        assert val_loader.batch_size == 3
        assert test_loader.batch_size == 3
        assert domain_loader.batch_size == 3

        # Check custom dataloader kwargs were applied
        assert train_loader.num_workers == 0
        assert val_loader.num_workers == 0
        assert test_loader.num_workers == 0
        assert domain_loader.num_workers == 0

    def test_domain_split_dataloaders_iteration(self, test_dataset):
        """Test that we can iterate through all created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1

        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=2,
            random_seed=42,
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        domain_batch = next(iter(domain_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch, domain_batch]:
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch

        # Verify domain_batch contains only the specified domain
        assert all(d != domain for d in train_batch["domains"])
        assert all(d != domain for d in val_batch["domains"])
        assert all(d != domain for d in test_batch["domains"])
        assert all(d == domain for d in domain_batch["domains"])

    def test_domain_split_dataloaders_raises_error_for_invalid_domain(
        self, test_dataset
    ):
        """Test that domain_split_dataloaders raises ValueError for invalid domain."""
        data = load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        # Use a domain that doesn't exist
        invalid_domain = 999

        # Should raise ValueError
        with pytest.raises(ValueError, match=r"Domain 999 has no associated data"):
            domain_split_dataloaders(
                data,
                train_keys,
                val_keys,
                test_keys,
                invalid_domain,
                batch_size=2,
                random_seed=42,
            )
