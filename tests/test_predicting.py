import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
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
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


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

    def test_save_chunk_creates_file(self, temp_output_dir):
        """Test that saving a chunk creates a file with correct naming."""
        output_path = temp_output_dir / "predictions.safetensors"
        chunk_predictions = {
            "file_001": torch.randn(10, 3),
            "file_002": torch.randn(10, 3),
        }

        _save_chunk_safetensors(chunk_predictions, output_path, chunk_idx=0)

        chunk_path = temp_output_dir / "predictions.chunk_0.safetensors"
        assert chunk_path.exists()

        # Verify contents
        loaded = load_file(chunk_path)
        assert "file_001" in loaded
        assert "file_002" in loaded


class TestCombineChunksSafetensors:
    """Tests for _combine_chunks_safetensors function."""

    def test_combine_zero_chunks(self, temp_output_dir):
        """Test that combining zero chunks does nothing."""
        output_path = temp_output_dir / "predictions.safetensors"
        _combine_chunks_safetensors(output_path, num_chunks=0)
        assert not output_path.exists()

    def test_combine_multiple_chunks(self, temp_output_dir):
        """Test combining multiple chunks into a single file."""
        output_path = temp_output_dir / "predictions.safetensors"

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
            chunk_path = temp_output_dir / f"predictions.chunk_{i}.safetensors"
            assert not chunk_path.exists()

        # Verify all predictions are in the combined file
        loaded = load_file(output_path)
        assert len(loaded) == 6  # 3 chunks x 2 files each
        for i in range(6):
            assert f"file_{i:03d}" in loaded


class TestSavePredictionsChunked:
    """Tests for save_predictions_chunked function."""

    def test_save_predictions_basic(self, mock_model, mock_dataloader, temp_output_dir):
        """Test basic functionality of saving predictions."""
        output_path = temp_output_dir / "predictions.safetensors"

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

    def test_model_eval_called(self, mock_model, mock_dataloader, temp_output_dir):
        """Test that model.eval() is called."""
        output_path = temp_output_dir / "predictions.safetensors"

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
        self, temp_output_dir, mock_model_config, mock_trainer_config
    ):
        """Test loading a model with valid weights."""
        # This test requires mocking create_model and the safetensors loading
        model_path = temp_output_dir / "model.safetensors"

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
            load_model_eval(model_path, mock_model_config, mock_trainer_config)

            # Verify create_model was called with correct args
            mock_create_model.assert_called_once_with(
                model_cfg=mock_model_config,
                trainer_cfg=mock_trainer_config,
            )

            # Verify load_state_dict was called
            mock_model.load_state_dict.assert_called_once_with(mock_state_dict)

            # Verify eval was called
            mock_model.eval.assert_called_once()

    def test_load_model_with_old_format(
        self, temp_output_dir, mock_model_config, mock_trainer_config
    ):
        """Test loading a model with old sincnet format."""
        model_path = temp_output_dir / "model.safetensors"

        with (
            patch("dr_sad.predicting.create_model") as mock_create_model,
            patch("dr_sad.predicting.load_file") as mock_load_file,
            patch("dr_sad.predicting.map_sincnet_weights") as mock_map_weights,
        ):
            # Create a mock model
            mock_model = MagicMock(spec=torch.nn.Module)
            mock_model.eval.return_value = mock_model

            # Make load_state_dict fail first time, succeed second time
            mock_model.load_state_dict.side_effect = [
                RuntimeError("Missing key(s) in state_dict: sincnet.features.0.weight"),
                None,
            ]

            mock_create_model.return_value = mock_model

            # Mock the state dict
            old_state_dict = {"old.weight": torch.randn(10, 10)}
            mapped_state_dict = {"sincnet.features.0.weight": torch.randn(10, 10)}
            mock_load_file.return_value = old_state_dict
            mock_map_weights.return_value = mapped_state_dict

            # Load the model
            load_model_eval(model_path, mock_model_config, mock_trainer_config)

            # Verify weight mapping was called
            mock_map_weights.assert_called_once_with(old_state_dict)

            # Verify load_state_dict was called twice (once failed, once succeeded)
            assert mock_model.load_state_dict.call_count == 2


class TestLoadDataEval:
    """Tests for load_data_eval function."""

    @pytest.fixture()
    def mock_data_split(self, temp_output_dir):
        """Create a mock data split file."""
        split_data = {
            "train": ["file_001", "file_002", "file_003"],
            "val": ["file_004", "file_005"],
            "test": ["file_006", "file_007"],
        }

        # Create temporary dataset directory structure
        dataset_dir = temp_output_dir / "test_dataset"
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

    def test_load_data_all_domains(
        self, mock_data_split, mock_configs, temp_output_dir
    ):
        """Test loading data with domain_type='all'."""
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs

        # Create the data directory structure
        data_dir = temp_output_dir / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)

        # Copy the split file to the expected location
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.from_keys_dataloaders") as mock_from_keys,
            patch("dr_sad.predicting.MAIN_DIR", temp_output_dir),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Mock the dataloaders
            mock_test_loader = MagicMock(spec=DataLoader)
            mock_from_keys.return_value = (None, None, mock_test_loader)

            # Load data
            test_loader, domain_loader = load_data_eval(
                data_cfg=data_cfg,
                data_split=None,
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
            assert test_loader is mock_test_loader
            assert domain_loader is None

    def test_load_data_exclude_one_domain(
        self, mock_data_split, mock_configs, temp_output_dir
    ):
        """Test loading data with domain_type='exclude_one'."""
        split_data, split_path = mock_data_split
        data_cfg, trainer_cfg, exp_config = mock_configs
        data_cfg["domain_type"] = "exclude_one"

        # Create the data directory structure
        data_dir = temp_output_dir / "data" / data_cfg["name"]
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_split_path = data_dir / data_cfg["split_name"]
        shutil.copy(split_path, dest_split_path)

        with (
            patch("dr_sad.predicting.load_data") as mock_load_data,
            patch("dr_sad.predicting.domain_split_dataloaders") as mock_domain_split,
            patch("dr_sad.predicting.MAIN_DIR", temp_output_dir),
        ):
            # Mock the data loading
            mock_dataset = MagicMock()
            mock_load_data.return_value = mock_dataset

            # Mock the dataloaders
            mock_test_loader = MagicMock(spec=DataLoader)
            mock_domain_loader = MagicMock(spec=DataLoader)
            mock_domain_split.return_value = (
                None,
                None,
                mock_test_loader,
                mock_domain_loader,
            )

            # Load data with excluded domain
            test_loader, domain_loader = load_data_eval(
                data_cfg=data_cfg,
                data_split=None,
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
            assert test_loader is mock_test_loader
            assert domain_loader is mock_domain_loader
