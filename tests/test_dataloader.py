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
    one_test_dataloader,
    single_domain_dataloaders,
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

    def test_load_example_dataset(self, example_dataset):
        """Test loading the test dataset using DrSadDataset."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

        assert isinstance(first_item["waveforms"], np.ndarray)
        assert first_item["waveforms"].ndim == 1  # Mono audio
        assert isinstance(first_item["waveforms"][0], np.float32 | np.float64)

        # Check that annotations are correct for the first file
        first_annotations = first_item["annotations"]
        assert len(first_annotations) == 2
        assert first_annotations[0] == (0.5, 1.0)

    def test_time_slice_functionality(self):
        """Test that time_slice correctly splits waveforms and adjusts annotations."""
        # Create a test DataFrame
        sample_rate = 16_000  # 16 kHz
        test_df = pd.DataFrame(
            {
                "file_ids": [
                    "file_01",
                    "file_02",
                    "file_03",
                ],
                "waveforms": [
                    0.3 * np.ones(round(1.6 * sample_rate)),  # 1.6 seconds
                    0.4 * np.ones(round(2.6 * sample_rate)),  # 2.6 seconds
                    0.5 * np.ones(round(0.6 * sample_rate)),  # 0.6 seconds
                ],
                "annotations": [
                    [(0.1, 1.1)],
                    [(0.4, 1.4), (1.6, 1.8)],
                    [(0.3, 0.4)],
                ],
                "domains": [0, 1, 0],
            }
        )
        test_df = test_df.set_index("file_ids")

        # Test with a time_slice of 1 second
        time_slice = 1.0  # seconds
        dataset = DrSadDataset(test_df, time_slice=time_slice)
        # Verify the number of segments created
        assert len(dataset) == 4  # 1 + 2 + 1 = 4 segments

        # Verify the first segment of the first file
        first_segment = dataset[0]
        assert first_segment["file_id"] == "file_01-00"
        assert len(first_segment["waveforms"]) == round(time_slice * sample_rate)
        np.testing.assert_array_almost_equal(first_segment["annotations"], [(0.0, 0.8)])
        assert first_segment["domains"] == 0

        # Verify the first segment of the second file
        second_segment = dataset[1]
        assert second_segment["file_id"] == "file_02-00"
        assert len(second_segment["waveforms"]) == round(time_slice * sample_rate)
        np.testing.assert_array_almost_equal(
            second_segment["annotations"], [(0.1, 1.0)]
        )
        assert second_segment["domains"] == 1

        # Verify the second segment of the second file
        third_segment = dataset[2]
        assert third_segment["file_id"] == "file_02-01"
        assert len(third_segment["waveforms"]) == round(time_slice * sample_rate)
        np.testing.assert_array_almost_equal(
            third_segment["annotations"], [(0.0, 0.1), (0.3, 0.5)]
        )
        assert third_segment["domains"] == 1

        # Verify the only segment of the third file
        fourth_segment = dataset[3]
        assert fourth_segment["file_id"] == "file_03-00"
        assert len(fourth_segment["waveforms"]) < round(time_slice * sample_rate)
        np.testing.assert_array_almost_equal(
            fourth_segment["annotations"], [(0.3, 0.4)]
        )
        assert fourth_segment["domains"] == 0

    def test_time_slice_bigger_dataset(self, example_dataset):
        """Test that time_slice works on a bigger dataset."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        time_slice = 1.0  # seconds
        dataset = DrSadDataset(data, time_slice=time_slice)

        # Check that the dataset length is greater than the original data length
        assert len(dataset) > len(data)

        # Check that all segments are of correct length
        for item in dataset:
            waveform_length = len(item["waveforms"])
            assert waveform_length <= int(time_slice * 16000)

    def test_from_dataset_split(self, example_dataset):
        """Test that DrSadDataset.train_test_split correctly splits the dataset."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_from_splitting_keys(self, example_dataset):
        """Test that DrSadDataset.from_splitting_keys correctly creates datasets."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_from_target_domain(self, example_dataset):
        """Test that DrSadDataset.from_target_domain correctly creates datasets."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        target_domain = 1

        train, val, test = DrSadDataset.from_target_domain(
            data, train_keys, val_keys, test_keys, target_domain
        )
        # Check that only data from the target domain is returned
        train_indices = set(train.data.index)
        val_indices = set(val.data.index)
        test_indices = set(test.data.index)

        # Check that splits are non-overlapping
        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)

        # Check that all returned datasets contain only target domain data
        for idx in train_indices:
            assert int(data.loc[idx, "domains"]) == target_domain
        for idx in val_indices:
            assert int(data.loc[idx, "domains"]) == target_domain
        for idx in test_indices:
            assert int(data.loc[idx, "domains"]) == target_domain

        # Check that returned datasets have the correct domain attribute
        assert train.domain == target_domain
        assert val.domain == target_domain
        assert test.domain == target_domain

        # Check that returned indices are subsets of provided keys filtered by domain
        domain_keys = set(data[data["domains"] == target_domain].index.tolist())
        expected_train = set(train_keys) & domain_keys
        expected_val = set(val_keys) & domain_keys
        expected_test = set(test_keys) & domain_keys

        assert train_indices == expected_train
        assert val_indices == expected_val
        assert test_indices == expected_test

    def test_from_target_domain_raises_error_for_invalid_domain(self, example_dataset):
        """Test that from_target_domain raises ValueError for domain with no data."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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
            DrSadDataset.from_target_domain(
                data, train_keys, val_keys, test_keys, invalid_domain
            )

    def test_from_split_domain(self, example_dataset):
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_from_split_domain_raises_error_for_invalid_domain(self, example_dataset):
        """Test that from_split_domain raises ValueError for domain with no data."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_time_slice_invalid_value(self):
        """Test that an invalid time_slice value raises a ValueError."""
        test_df = pd.DataFrame(
            {
                "waveforms": [np.arange(32000)],
                "annotations": [[(0.5, 1.5)]],
                "domains": [0],
            }
        )

        with pytest.raises(ValueError, match="time_slice must be a positive"):
            DrSadDataset(test_df, time_slice=-1.0)


class TestDataloader:
    def test_make_dataloader(self, example_dataset):
        """Test that make_dataloader creates a properly configured DataLoader."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_make_dataloader_with_custom_params(self, example_dataset):
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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
    def test_train_test_split_dataloaders_basic(self, example_dataset):
        """Test basic functionality of train_test_split_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_train_test_split_dataloaders_custom_params(self, example_dataset):
        """Test train_test_split_dataloaders with custom parameters."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_train_test_split_dataloaders_iteration(self, example_dataset):
        """Test that we can iterate through the created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_train_test_split_dataloaders_time_slice(self, example_dataset):
        """Test train_test_split_dataloaders with time_slice parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        time_slice = 1.0  # seconds

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=2,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            time_slice=time_slice,
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch]:
            assert "file_id" in batch
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2
            assert batch["waveforms"].shape[1] <= int(time_slice * 16000)
            for anno in batch["annotations"]:
                for start, end in anno:
                    assert start >= 0.0
                    assert end <= time_slice


class TestFromKeysDataloaders:
    def test_from_keys_dataloaders_basic(self, example_dataset):
        """Test basic functionality of from_keys_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_from_keys_dataloaders_iteration(self, example_dataset):
        """Test that we can iterate through the created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_from_keys_dataloaders_time_slice(self, example_dataset):
        """Test from_keys_dataloaders with time_slice parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        time_slice = 1.0  # seconds

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            batch_size=2,
            random_seed=42,
            time_slice=time_slice,
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch]:
            assert "file_id" in batch
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2
            assert batch["waveforms"].shape[1] <= int(time_slice * 16000)
            for anno in batch["annotations"]:
                for start, end in anno:
                    assert start >= 0.0
                    assert end <= time_slice


class TestDomainSplitDataloaders:
    def test_domain_split_dataloaders_basic(self, example_dataset):
        """Test basic functionality of domain_split_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_domain_split_dataloaders_custom_params(self, example_dataset):
        """Test domain_split_dataloaders with custom parameters."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_domain_split_dataloaders_iteration(self, example_dataset):
        """Test that we can iterate through all created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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
        self, example_dataset
    ):
        """Test that domain_split_dataloaders raises ValueError for invalid domain."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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

    def test_domain_split_dataloaders_time_slice(self, example_dataset):
        """Test domain_split_dataloaders with time_slice parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1

        time_slice = 1.0  # seconds

        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=2,
            random_seed=42,
            time_slice=time_slice,
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        domain_batch = next(iter(domain_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch, domain_batch]:
            assert "file_id" in batch
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2
            assert batch["waveforms"].shape[1] <= int(time_slice * 16000)
            for anno in batch["annotations"]:
                for start, end in anno:
                    assert start >= 0.0
                    assert end <= time_slice


class TestSingleDomainDataloaders:
    def test_single_domain_dataloaders_basic(self, example_dataset):
        """Test basic functionality of single_domain_dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
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

        # Check that all have StratifiedSampler
        assert isinstance(train_loader.sampler, StratifiedSampler)
        assert isinstance(val_loader.sampler, StratifiedSampler)
        assert isinstance(test_loader.sampler, StratifiedSampler)

    def test_single_domain_dataloaders_custom_params(self, example_dataset):
        """Test single_domain_dataloaders with custom parameters."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1
        custom_kwargs = {"num_workers": 0}

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
            batch_size=3,
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

    def test_single_domain_dataloaders_iteration(self, example_dataset):
        """Test that we can iterate through the created dataloaders."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
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

        # Verify all batches contain only the target domain
        assert all(d == target_domain for d in train_batch["domains"])
        assert all(d == target_domain for d in val_batch["domains"])
        assert all(d == target_domain for d in test_batch["domains"])

    def test_single_domain_dataloaders_raises_error_for_invalid_domain(
        self, example_dataset
    ):
        """Test that single_domain_dataloaders raises ValueError for invalid domain."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
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
            single_domain_dataloaders(
                data,
                train_keys,
                val_keys,
                test_keys,
                invalid_domain,
                batch_size=2,
                random_seed=42,
            )

    def test_single_domain_dataloaders_time_slice(self, example_dataset):
        """Test single_domain_dataloaders with time_slice parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1

        time_slice = 1.0  # seconds

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
            batch_size=2,
            random_seed=42,
            time_slice=time_slice,
        )

        # Test that we can get batches from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch]:
            assert "file_id" in batch
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2
            assert batch["waveforms"].shape[1] <= int(time_slice * 16000)
            for anno in batch["annotations"]:
                for start, end in anno:
                    assert start >= 0.0
                    assert end <= time_slice

        # Verify all batches contain only the target domain
        assert all(d == target_domain for d in train_batch["domains"])
        assert all(d == target_domain for d in val_batch["domains"])
        assert all(d == target_domain for d in test_batch["domains"])


class TestOneTestDataloader:
    def test_one_test_dataloader_basic(self, example_dataset):
        """Test basic functionality of one_test_dataloader."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        test_loader = one_test_dataloader(
            data,
            data_keys=data.index.tolist()[:6],
            batch_size=2,
        )

        # Check that returned object is DataLoader
        assert isinstance(test_loader, DataLoader)

        # Check batch size
        assert test_loader.batch_size == 2

        assert len(test_loader) == 3

        for batch in test_loader:
            assert "file_id" in batch
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch

    def test_one_test_dataloader_time_slice(self, example_dataset):
        """Test one_test_dataloader with time_slice parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        time_slice = 1.0  # seconds

        test_loader = one_test_dataloader(
            data,
            data_keys=data.index.tolist()[:6],
            batch_size=2,
            time_slice=time_slice,
        )

        # Test that we can get batches from the loader
        test_batch = next(iter(test_loader))

        assert len(test_loader) == 3 * int(2.5 / time_slice)

        # Check batch structure
        assert "file_id" in test_batch
        assert "waveforms" in test_batch
        assert "annotations" in test_batch
        assert "domains" in test_batch
        assert len(test_batch["waveforms"]) == 2
        assert test_batch["waveforms"].shape[1] <= int(time_slice * 16000)
        for anno in test_batch["annotations"]:
            for start, end in anno:
                assert start >= 0.0
                assert end <= time_slice


class TestNoiseAugmentation:
    def test_load_with_noise_and_without(self, example_dataset, noise_dataset):
        """Loading the same data with and without noise should change waveforms."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Create two independent copies so the augmentation doesn't modify the
        # other one in-place.
        data_no_noise = data.copy(deep=True)
        data_with_noise = data.copy(deep=True)

        # Prepare noise kwargs pointing to the generated noise folder in the
        # fixture (`noise_dataset` returns the tmp path containing `noise`).
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
        }

        ds_no = DrSadDataset(data_no_noise, sample_rate=16000)
        ds_noise = DrSadDataset(
            data_with_noise, sample_rate=16000, noise_kwargs=noise_kwargs
        )

        for datum_no, datum_noise in zip(ds_no, ds_noise, strict=True):
            assert datum_no["file_id"] == datum_noise["file_id"]
            assert datum_no["domains"] == datum_noise["domains"]
            assert datum_no["annotations"] == datum_noise["annotations"]
            # Waveforms should differ due to noise addition
            assert not np.array_equal(datum_no["waveforms"], datum_noise["waveforms"])
            assert datum_noise["waveforms"].shape == datum_no["waveforms"].shape

    def test_from_target_domain_with_noise(self, example_dataset, noise_dataset):
        """Ensure from_target_domain accepts noise_kwargs and modifies them."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        target_domain = 1

        # Prepare noise kwargs pointing to the generated noise folder in the fixture
        noise_dir = noise_dataset / "noise"
        base_seed = 42
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train, val, test = DrSadDataset.from_target_domain(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
            time_slice=None,
            sample_rate=16000,
            noise_kwargs=noise_kwargs,
        )

        # Check noise kwargs were set on returned datasets
        assert train.noise_kwargs is not None
        assert val.noise_kwargs is not None
        assert test.noise_kwargs is not None

        # Seeds should have been incremented for each split
        assert train.noise_kwargs["seed"] == base_seed + 1
        assert val.noise_kwargs["seed"] == base_seed + 2
        assert test.noise_kwargs["seed"] == base_seed + 3

        # noise_dir should be preserved
        assert train.noise_kwargs["noise_dir"] == noise_dir
        assert val.noise_kwargs["noise_dir"] == noise_dir
        assert test.noise_kwargs["noise_dir"] == noise_dir

        # Basic sanity: datasets should be non-empty and carry the domain attribute
        assert train.domain == target_domain
        assert val.domain == target_domain
        assert test.domain == target_domain
        assert len(train) + len(val) + len(test) == len(
            data[data["domains"] == target_domain]
        )

    def test_from_split_domain_with_noise(self, example_dataset, noise_dataset):
        """Test that from_split_domain accepts noise_kwargs and propagates them."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        domain = 1
        base_seed = 42

        # Prepare noise kwargs
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train, val, test, domain_data = DrSadDataset.from_split_domain(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            sample_rate=16000,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were set on all datasets
        assert train.noise_kwargs is not None
        assert val.noise_kwargs is not None
        assert test.noise_kwargs is not None
        assert domain_data.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train.noise_kwargs["seed"] == base_seed + 1
        assert val.noise_kwargs["seed"] == base_seed + 2
        assert test.noise_kwargs["seed"] == base_seed + 3
        assert domain_data.noise_kwargs["seed"] == base_seed + 4

        # Verify all datasets have the domain attribute
        assert train.domain == domain
        assert val.domain == domain
        assert test.domain == domain
        assert domain_data.domain == domain

        # Verify domain_data contains only the specified domain
        for idx in domain_data.data.index:
            assert int(data.loc[idx, "domains"]) == domain

        # Verify train/val/test exclude the domain
        for idx in train.data.index:
            assert int(data.loc[idx, "domains"]) != domain
        for idx in val.data.index:
            assert int(data.loc[idx, "domains"]) != domain
        for idx in test.data.index:
            assert int(data.loc[idx, "domains"]) != domain

    def test_from_splitting_keys_with_noise(self, example_dataset, noise_dataset):
        """Test that from_splitting_keys accepts noise_kwargs and propagates them."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Define splitting keys
        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        base_seed = 42

        # Prepare noise kwargs
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train, val, test = DrSadDataset.from_splitting_keys(
            data,
            train_keys,
            val_keys,
            test_keys,
            sample_rate=16000,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were set on all datasets
        assert train.noise_kwargs is not None
        assert val.noise_kwargs is not None
        assert test.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train.noise_kwargs["seed"] == base_seed + 1
        assert val.noise_kwargs["seed"] == base_seed + 2
        assert test.noise_kwargs["seed"] == base_seed + 3

        # Verify noise_dir is preserved
        assert train.noise_kwargs["noise_dir"] == noise_dir
        assert val.noise_kwargs["noise_dir"] == noise_dir
        assert test.noise_kwargs["noise_dir"] == noise_dir

        # Verify datasets contain the correct data
        assert len(train) == len(train_keys)
        assert len(val) == len(val_keys)
        assert len(test) == len(test_keys)

    def test_from_train_test_split_with_noise(self, example_dataset, noise_dataset):
        """Test that from_train_test_split accepts noise_kwargs and propagates them."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        base_seed = 42

        # Prepare noise kwargs
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train, val, test = DrSadDataset.from_train_test_split(
            data,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=123,
            sample_rate=16000,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were set on all datasets
        assert train.noise_kwargs is not None
        assert val.noise_kwargs is not None
        assert test.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train.noise_kwargs["seed"] == base_seed + 1
        assert val.noise_kwargs["seed"] == base_seed + 2
        assert test.noise_kwargs["seed"] == base_seed + 3

        # Verify noise_dir is preserved
        assert train.noise_kwargs["noise_dir"] == noise_dir
        assert val.noise_kwargs["noise_dir"] == noise_dir
        assert test.noise_kwargs["noise_dir"] == noise_dir

        # Verify datasets are non-empty and non-overlapping
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        train_indices = set(train.data.index)
        val_indices = set(val.data.index)
        test_indices = set(test.data.index)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)


class TestDataloadersWithNoise:
    def test_train_test_split_dataloaders_with_noise(
        self, example_dataset, noise_dataset
    ):
        """Test train_test_split_dataloaders with noise_kwargs parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        base_seed = 42

        # Prepare noise kwargs
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=2,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Check that all returned objects are DataLoaders
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Verify noise_kwargs were propagated to datasets
        assert train_loader.dataset.noise_kwargs is not None
        assert val_loader.dataset.noise_kwargs is not None
        assert test_loader.dataset.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train_loader.dataset.noise_kwargs["seed"] == base_seed + 1
        assert val_loader.dataset.noise_kwargs["seed"] == base_seed + 2
        assert test_loader.dataset.noise_kwargs["seed"] == base_seed + 3

        # Verify noise_dir is preserved
        assert train_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert val_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert test_loader.dataset.noise_kwargs["noise_dir"] == noise_dir

        # Test that we can iterate through batches without errors
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Check batch structure for each
        for batch in [train_batch, val_batch, test_batch]:
            assert "waveforms" in batch
            assert "annotations" in batch
            assert "domains" in batch
            assert len(batch["waveforms"]) == 2

    def test_from_keys_dataloaders_with_noise(self, example_dataset, noise_dataset):
        """Test from_keys_dataloaders with noise_kwargs parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        base_seed = 42
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to datasets
        assert train_loader.dataset.noise_kwargs is not None
        assert val_loader.dataset.noise_kwargs is not None
        assert test_loader.dataset.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train_loader.dataset.noise_kwargs["seed"] == base_seed + 1
        assert val_loader.dataset.noise_kwargs["seed"] == base_seed + 2
        assert test_loader.dataset.noise_kwargs["seed"] == base_seed + 3

    def test_domain_split_dataloaders_with_noise(self, example_dataset, noise_dataset):
        """Test domain_split_dataloaders with noise_kwargs parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1

        base_seed = 42
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        (
            train_loader,
            val_loader,
            test_loader,
            domain_loader,
        ) = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to all datasets
        assert train_loader.dataset.noise_kwargs is not None
        assert val_loader.dataset.noise_kwargs is not None
        assert test_loader.dataset.noise_kwargs is not None
        assert domain_loader.dataset.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train_loader.dataset.noise_kwargs["seed"] == base_seed + 1
        assert val_loader.dataset.noise_kwargs["seed"] == base_seed + 2
        assert test_loader.dataset.noise_kwargs["seed"] == base_seed + 3
        assert domain_loader.dataset.noise_kwargs["seed"] == base_seed + 4

    def test_single_domain_dataloaders_with_noise(self, example_dataset, noise_dataset):
        """Test single_domain_dataloaders with noise_kwargs parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1

        base_seed = 42
        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": base_seed,
            "start_choice": True,
        }

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to datasets
        assert train_loader.dataset.noise_kwargs is not None
        assert val_loader.dataset.noise_kwargs is not None
        assert test_loader.dataset.noise_kwargs is not None

        # Verify seeds were incremented for each split
        assert train_loader.dataset.noise_kwargs["seed"] == base_seed + 1
        assert val_loader.dataset.noise_kwargs["seed"] == base_seed + 2
        assert test_loader.dataset.noise_kwargs["seed"] == base_seed + 3

    def test_one_test_dataloader_with_noise(self, example_dataset, noise_dataset):
        """Test one_test_dataloader with noise_kwargs parameter."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        noise_dir = noise_dataset / "noise"
        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
        }

        test_loader = one_test_dataloader(
            data,
            data_keys=data.index.tolist()[:6],
            batch_size=2,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to dataset
        assert test_loader.dataset.noise_kwargs is not None
        assert test_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert test_loader.dataset.noise_kwargs["seed"] == 42

        # Test that we can iterate through batches without errors
        test_batch = next(iter(test_loader))
        assert "waveforms" in test_batch
        assert "annotations" in test_batch
        assert "domains" in test_batch

    def test_train_test_split_dataloaders_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """
        Test train_test_split_dataloaders with noise_kwargs that only apply to train.
        """
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
            "train_only": True,
        }

        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=2,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to train dataset but not val/test
        assert train_loader.dataset.noise_kwargs is not None
        assert train_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert train_loader.dataset.noise_kwargs["seed"] == noise_kwargs["seed"]
        assert val_loader.dataset.noise_kwargs is None
        assert test_loader.dataset.noise_kwargs is None

    def test_domain_split_dataloaders_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """
        Test domain_split_dataloaders with noise_kwargs that only apply to the train
        split.
        """
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        domain = 1

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
            "train_only": True,
        }

        (
            train_loader,
            val_loader,
            test_loader,
            domain_loader,
        ) = domain_split_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            domain,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to domain dataset but not train/val/test
        assert train_loader.dataset.noise_kwargs is not None
        assert train_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert train_loader.dataset.noise_kwargs["seed"] == noise_kwargs["seed"]
        assert domain_loader.dataset.noise_kwargs is None
        assert val_loader.dataset.noise_kwargs is None
        assert test_loader.dataset.noise_kwargs is None

    def test_single_domain_dataloaders_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """
        Test single_domain_dataloaders with noise_kwargs that only apply to the train.
        """
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()
        target_domain = 1

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
            "train_only": True,
        }

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            target_domain,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to datasets but only affect target domain
        assert train_loader.dataset.noise_kwargs is not None
        assert train_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert train_loader.dataset.noise_kwargs["seed"] == noise_kwargs["seed"]
        assert val_loader.dataset.noise_kwargs is None
        assert test_loader.dataset.noise_kwargs is None

    def test_from_keys_dataloaders_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """Test from_keys_dataloaders with noise_kwargs where train_only is True."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        train_keys = data.index[:12].tolist()
        val_keys = data.index[12:16].tolist()
        test_keys = data.index[16:].tolist()

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "start_choice": True,
            "train_only": True,
        }

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys,
            val_keys,
            test_keys,
            batch_size=2,
            random_seed=42,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were set on train dataset but not val/test
        assert train_loader.dataset.noise_kwargs is not None
        assert train_loader.dataset.noise_kwargs["noise_dir"] == noise_dir
        assert train_loader.dataset.noise_kwargs["seed"] == noise_kwargs["seed"]
        assert val_loader.dataset.noise_kwargs is None
        assert test_loader.dataset.noise_kwargs is None

    def test_from_target_domain_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """Test from_target_domain with noise_kwargs where train_only is True."""
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "train_only": True,
        }

        train, val, test = DrSadDataset.from_target_domain(
            data,
            train_keys=data.index[:12].tolist(),
            val_keys=data.index[12:16].tolist(),
            test_keys=data.index[16:].tolist(),
            domain=1,
            time_slice=None,
            sample_rate=16000,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were set on train dataset but not val/test
        assert train.noise_kwargs is not None
        assert train.noise_kwargs["noise_dir"] == noise_dir
        assert train.noise_kwargs["seed"] == noise_kwargs["seed"]
        assert val.noise_kwargs is None
        assert test.noise_kwargs is None

    def test_one_test_dataloader_with_train_only_noise(
        self,
        example_dataset,
        noise_dataset,
    ):
        """
        Test one_test_dataloader with noise_kwargs where train_only is True.
        """
        data = load_data(
            data_choice=None,
            data_set_path=example_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        noise_dir = noise_dataset / "noise"

        noise_kwargs = {
            "noise_dir": noise_dir,
            "snr_db": 10.0,
            "simultaneous": 1,
            "seed": 42,
            "train_only": True,
        }

        test_loader = one_test_dataloader(
            data,
            data_keys=data.index.tolist()[:6],
            batch_size=2,
            noise_kwargs=noise_kwargs,
        )

        # Verify noise_kwargs were propagated to dataset
        assert test_loader.dataset.noise_kwargs is None
