import shutil
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from dr_sad.predicting import (
    _combine_chunks_safetensors,
    _save_chunk_safetensors,
    load_data_eval,
    load_model_eval,
    save_predictions_chunked,
)


@pytest.fixture()
def mock_model():
    """Create a mock PyTorch model for testing."""
    model = MagicMock(spec=torch.nn.Module)
    model.eval.return_value = model

    # Mock predict_step to return tensors
    def predict_step(batch, _batch_idx):
        batch_size = len(batch["file_id"])
        # Return a tensor with shape (batch_size, 10, 3) as typical output
        return torch.randn(batch_size, 10, 3)

    model.predict_step = predict_step
    return model


@pytest.fixture()
def mock_dataloader():
    """Create a mock DataLoader for testing."""
    # Create 3 batches with 2 samples each
    batches = [
        {"file_id": ["file_001", "file_002"]},
        {"file_id": ["file_003", "file_004"]},
        {"file_id": ["file_005", "file_006"]},
    ]

    class MockDataLoader:
        def __init__(self, batches: list[dict[str, list[str]]]) -> None:
            self.batches = batches

        def __iter__(self) -> Iterator[dict[str, list[str]]]:
            return iter(self.batches)

        def __len__(self) -> int:
            return len(self.batches)

    return MockDataLoader(batches)


class TestSaveChunkSafetensors:
    """Tests for _save_chunk_safetensors function."""

    def test_save_chunk_creates_file(self, tmp_path):
        """Test that saving a chunk creates a file with correct naming."""
        output_path = tmp_path / "predictions.safetensors"
        chunk_predictions = {
            "file_001": torch.randn(10, 3),
            "file_002": torch.randn(10, 3),
        }

        _save_chunk_safetensors(chunk_predictions, output_path, chunk_idx=0)
        chunk_path = tmp_path / "predictions.chunk_0.safetensors"
        assert chunk_path.exists()
        # Verify contents
        loaded = load_file(chunk_path)
        assert "file_001" in loaded
        assert "file_002" in loaded


class TestCombineChunksSafetensors:
    """Tests for _combine_chunks_safetensors function."""

    def test_combine_zero_chunks(self, tmp_path):
        """Test that combining zero chunks does nothing and raises error."""
        output_path = tmp_path / "predictions.safetensors"
        with pytest.raises(ValueError, match=r"No chunks to combine."):
            _combine_chunks_safetensors(output_path, num_chunks=0)
        assert not output_path.exists()

    def test_combine_multiple_chunks(self, tmp_path):
        """Test combining multiple chunks into a single file."""
        output_path = tmp_path / "predictions.safetensors"

        # Create multiple chunks
        for i in range(3):
            chunk_predictions = {
                f"file_{i * 2:03d}": torch.randn(10, 3),
                f"file_{i * 2 + 1:03d}": torch.randn(10, 3),
            }
            _save_chunk_safetensors(chunk_predictions, output_path, chunk_idx=i)

        # Combine all chunks
        _combine_chunks_safetensors(output_path, num_chunks=3)

        # Verify combined file exists and chunks are deleted
        assert output_path.exists()
        for i in range(3):
            chunk_path = tmp_path / f"predictions.chunk_{i}.safetensors"
            assert not chunk_path.exists()

        # Verify all predictions are in the combined file
        loaded = load_file(output_path)
        assert len(loaded) == 6  # 3 chunks x 2 files each
        for i in range(6):
            assert f"file_{i:03d}" in loaded


class TestSavePredictionsChunked:
    """Tests for save_predictions_chunked function."""

    def test_save_predictions_basic(self, mock_model, mock_dataloader, tmp_path):
        """Test basic functionality of saving predictions."""
        output_path = tmp_path / "predictions.safetensors"

        save_predictions_chunked(
            mock_model, mock_dataloader, output_path, chunk_size=10
        )

        # Check that final file exists
        assert output_path.exists()

        # Verify contents
        loaded = load_file(output_path)
        assert len(loaded) == 6  # 3 batches x 2 files each
        for i in range(1, 7):
            assert f"file_{i:03d}" in loaded
            # Check shape is preserved
            assert loaded[f"file_{i:03d}"].shape == (10, 3)

    def test_model_eval_called(self, mock_model, mock_dataloader, tmp_path):
        """Test that model.eval() is called."""
        output_path = tmp_path / "predictions.safetensors"

        save_predictions_chunked(
            mock_model, mock_dataloader, output_path, chunk_size=10
        )

        mock_model.eval.assert_called_once()


class TestLoadModelEval:
    """Tests for load_model_eval function."""

    @pytest.fixture()
    def mock_model_config(self):
        """Create a mock model configuration."""
        return {
            "sincnet_stride": 10,
            "sincnet_sample_rate": 16000,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "linear_hidden_size": 128,
        }

    @pytest.fixture()
    def mock_trainer_config(self):
        """Create a mock trainer configuration."""
        return {
            "lr": 0.001,
        }

    def test_load_model_with_valid_weights(
        self, tmp_path, mock_model_config, mock_trainer_config
    ):
        """Test loading a model with valid weights."""
        # This test requires mocking create_model and the safetensors loading
        model_path = tmp_path / "model.safetensors"

        with (
            patch("dr_sad.predicting.create_model") as mock_create_model,
            patch("dr_sad.predicting.load_file") as mock_load_file,
        ):
            # Create a mock model
            mock_model = MagicMock(spec=torch.nn.Module)
            mock_model.eval.return_value = mock_model
            mock_create_model.return_value = mock_model

            # Mock the state dict
            mock_state_dict = {"layer.weight": torch.randn(10, 10)}
            mock_load_file.return_value = mock_state_dict

            # Load the model
            result_model = load_model_eval(
                model_path, mock_model_config, mock_trainer_config
            )

            # Verify create_model was called with correct args
            mock_create_model.assert_called_once_with(
                model_cfg=mock_model_config,
                trainer_cfg=mock_trainer_config,
                data_cfg=None,
            )

            # Verify load_file was called with the model path
            mock_load_file.assert_called_once_with(model_path)

            # Verify load_state_dict was called
            mock_model.load_state_dict.assert_called_once_with(mock_state_dict)

            # Verify eval was called
            mock_model.eval.assert_called_once()

            # Verify the returned model is the mocked model
            assert result_model is mock_model

    def test_load_model_with_incorrect_format(
        self, tmp_path, mock_model_config, mock_trainer_config
    ):
        """Test that loading a model with mismatched state dict raises an error."""
        model_path = tmp_path / "model.safetensors"

        with (
            patch("dr_sad.predicting.create_model") as mock_create_model,
            patch("dr_sad.predicting.load_file") as mock_load_file,
        ):
            # Create a mock model with expected keys
            mock_model = MagicMock(spec=torch.nn.Module)
            mock_model.eval.return_value = mock_model
            # Mock state_dict to return expected keys
            mock_model.state_dict.return_value = {
                "expected_layer1.weight": torch.randn(10, 10),
                "expected_layer2.bias": torch.randn(10),
            }
            mock_create_model.return_value = mock_model

            # Mock the state dict with wrong keys (missing some, extra others)
            mock_state_dict = {
                "wrong_key.weight": torch.randn(10, 10),
                "another_wrong_key.bias": torch.randn(10),
            }
            mock_load_file.return_value = mock_state_dict

            # Configure load_state_dict to raise error for mismatched keys
            def raise_on_load(_state_dict, strict=True):
                if strict:
                    msg = (
                        "Error(s) in loading state_dict for Model:\n"
                        "Missing key(s): expected_layer1.weight, expected_layer2.bias\n"
                        "Unexpected key(s): wrong_key.weight, another_wrong_key.bias"
                    )
                    raise RuntimeError(msg)

            mock_model.load_state_dict.side_effect = raise_on_load

            # Verify that loading raises the RuntimeError
            with pytest.raises(RuntimeError, match=r"Error.*in loading state_dict"):
                load_model_eval(model_path, mock_model_config, mock_trainer_config)


class TestLoadDataEval:
    """Tests for load_data_eval function."""

    @pytest.fixture()
    def mock_data_split(self, tmp_path):
        """Create a mock data split file."""
        split_data = {
            "train": ["file_001", "file_002", "file_003"],
            "val": ["file_004", "file_005"],
            "test": ["file_006", "file_007"],
        }

        # Create temporary dataset directory structure
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        split_path = dataset_dir / "datasplit.yaml"
        with open(split_path, "w") as f:
            yaml.dump(split_data, f)
        return split_data, split_path

    @pytest.fixture()
    def mock_configs(self):
        """Create mock configuration dictionaries."""
        data_cfg = {
            "name": "test_dataset",
            "split_name": "datasplit.yaml",
            "domain_type": "all",
        }
        trainer_cfg = {
            "num_workers": 4,
            "batch_size": 16,
        }
        exp_config = {
            "random_seed": 42,
        }
        return data_cfg, trainer_cfg, exp_config

    def test_load_data_all_domains(self, mock_data_split, mock_configs, tmp_path):
        """Test loading data with domain_type='all'."""
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs

        # Create the data directory structure
        data_dir = tmp_path / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)

        # Copy the split file to the expected location
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.from_keys_dataloaders") as mock_from_keys,
            patch("dr_sad.predicting.MAIN_DIR", tmp_path),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Mock the dataloaders
            mock_val_loader = MagicMock(spec=DataLoader)
            mock_test_loader = MagicMock(spec=DataLoader)
            mock_from_keys.return_value = (None, mock_val_loader, mock_test_loader)

            # Load data
            validation_loader, test_loader, domain_loader = load_data_eval(
                data_cfg=data_cfg,
                data_split=split_data,
                trainer_cfg=trainer_cfg,
                exp_config=exp_config,
                exclude_domain=None,
            )

            # Verify load_data was called correctly
            mock_load_data.assert_called_once_with("test_dataset", num_workers=4)

            # Verify from_keys_dataloaders was called with correct arguments
            mock_from_keys.assert_called_once_with(
                mock_dataset,
                train_keys=split_data["train"],
                val_keys=split_data["val"],
                test_keys=split_data["test"],
                batch_size=16,
                random_seed=42,
            )

            # Verify return values
            assert validation_loader is mock_val_loader
            assert test_loader is mock_test_loader
            assert domain_loader is None

    def test_load_data_exclude_one_domain(
        self, mock_data_split, mock_configs, tmp_path
    ):
        """Test loading data with domain_type='exclude_one'."""
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs
        data_cfg["domain_type"] = "exclude_one"

        # Create the data directory structure
        data_dir = tmp_path / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.domain_split_dataloaders") as mock_domain_split,
            patch("dr_sad.predicting.MAIN_DIR", tmp_path),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Mock the dataloaders
            mock_val_loader = MagicMock(spec=DataLoader)
            mock_test_loader = MagicMock(spec=DataLoader)
            mock_domain_loader = MagicMock(spec=DataLoader)
            mock_domain_split.return_value = (
                None,
                mock_val_loader,
                mock_test_loader,
                mock_domain_loader,
            )

            # Load data with excluded domain
            validation_loader, test_loader, domain_loader = load_data_eval(
                data_cfg=data_cfg,
                data_split=split_data,
                trainer_cfg=trainer_cfg,
                exp_config=exp_config,
                exclude_domain=0,
            )

            # Verify domain_split_dataloaders was called with correct arguments
            mock_domain_split.assert_called_once_with(
                mock_dataset,
                train_keys=split_data["train"],
                val_keys=split_data["val"],
                test_keys=split_data["test"],
                domain=0,
                batch_size=16,
                random_seed=42,
            )

            # Verify return values
            assert validation_loader is mock_val_loader
            assert test_loader is mock_test_loader
            assert domain_loader is mock_domain_loader

    def test_load_data_single_domain(self, mock_data_split, mock_configs, tmp_path):
        """Test loading data with domain_type='single_domain'."""
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs
        data_cfg["domain_type"] = "single_domain"

        # Create the data directory structure
        data_dir = tmp_path / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.single_domain_dataloaders") as mock_single_domain,
            patch("dr_sad.predicting.MAIN_DIR", tmp_path),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Mock the dataloaders
            mock_val_loader = MagicMock(spec=DataLoader)
            mock_test_loader = MagicMock(spec=DataLoader)
            mock_single_domain.return_value = (None, mock_val_loader, mock_test_loader)

            # Load data with train_domain specified
            validation_loader, test_loader, domain_loader = load_data_eval(
                data_cfg=data_cfg,
                data_split=split_data,
                trainer_cfg=trainer_cfg,
                exp_config=exp_config,
                train_domain=2,
            )

            # Verify load_data was called correctly
            mock_load_data.assert_called_once_with("test_dataset", num_workers=4)

            # Verify single_domain_dataloaders was called with correct arguments
            mock_single_domain.assert_called_once_with(
                mock_dataset,
                train_keys=split_data["train"],
                val_keys=split_data["val"],
                test_keys=split_data["test"],
                domain=2,
                batch_size=16,
                random_seed=42,
            )

            # Verify return values
            assert validation_loader is mock_val_loader
            assert test_loader is mock_test_loader
            assert domain_loader is None

    def test_load_data_single_domain_missing_train_domain(
        self, mock_data_split, mock_configs, tmp_path
    ):
        """
        Test that load_data_eval raises error when train_domain is
        missing for single_domain.
        """
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs
        data_cfg["domain_type"] = "single_domain"

        # Create the data directory structure
        data_dir = tmp_path / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.MAIN_DIR", tmp_path),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Verify that ValueError is raised when train_domain is None
            with pytest.raises(
                ValueError,
                match=(
                    r"Must specify --train_domain when domain_type is "
                    r"'single_domain'\."
                ),
            ):
                load_data_eval(
                    data_cfg=data_cfg,
                    data_split=split_data,
                    trainer_cfg=trainer_cfg,
                    exp_config=exp_config,
                    train_domain=None,
                )

    def test_load_data_all_with_train_domain_error(
        self, mock_data_split, mock_configs, tmp_path
    ):
        """
        Test that load_data_eval raises error when train_domain is provided for 'all'
        domain_type.
        """
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs
        # domain_type is already 'all' by default

        # Create the data directory structure
        data_dir = tmp_path / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.MAIN_DIR", tmp_path),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Verify that ValueError is raised when train_domain is provided for 'all'
            with pytest.raises(
                ValueError,
                match=r"Cannot specify train_domain when domain_type is set to 'all'\.",
            ):
                load_data_eval(
                    data_cfg=data_cfg,
                    data_split=split_data,
                    trainer_cfg=trainer_cfg,
                    exp_config=exp_config,
                    train_domain=1,
                )
