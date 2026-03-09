import warnings

import numpy as np
import pytest
import torch

from dr_sad.pyannet import PyanNet


class TestPyanNet:
    def test_instantiation_and_basic_props(self):
        p = PyanNet()
        # basic properties
        assert isinstance(p.sample_rate, int)
        assert p.sample_rate == 16000

        num_frames = p.num_frames(16000)
        assert isinstance(num_frames, int)
        assert num_frames > 0

        rf_size = p.receptive_field_size(1)
        assert isinstance(rf_size, int)
        assert rf_size > 0

        center = p.receptive_field_center(0)
        assert isinstance(center, int)

    def test_forward_and_output_shape(self):
        p = PyanNet()
        batch = 2
        samples = 16000
        x = torch.randn(batch, 1, samples)
        # run forward on CPU without grad
        with torch.no_grad():
            out = p(x)

        # Expect shape (batch, frames, classes)
        assert out.ndim == 3
        assert out.shape[0] == batch
        assert out.shape[2] == p.num_classes

    def test_frame_centers_and_start_step(self):
        p = PyanNet()
        start, step = p.frame_centers_start_step()
        assert isinstance(start, float)
        assert isinstance(step, float)

        centers = p.frame_centers(16000, as_numpy=True)
        assert isinstance(centers, np.ndarray)
        assert centers.shape[0] == p.num_frames(16000)
        # centers should be strictly increasing
        assert np.all(np.diff(centers) > 0)

    def test_prepare_annotation_shape(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        assert isinstance(speaker_truth, torch.Tensor)
        # (batch, 1, frames)
        assert speaker_truth.shape[0] == 1
        assert speaker_truth.shape[1] == 1
        assert speaker_truth.shape[2] == p.num_frames(waveforms.shape[-1])
        assert speaker_truth[0, 0, 2] == 1.0
        assert speaker_truth[0, 0, -2] == 0.0

    def test_loss_function(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)
        loss = p.loss_function(speaker_truth, [0], outputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_accuracy_function(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)
        acc = p.accuracy_function(speaker_truth, [0], outputs, threshold=0.5)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_configure_optimizers_no_scheduler(self):
        """Test that configure_optimizers returns optimizer only when no scheduler."""
        p = PyanNet()
        result = p.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)

    def test_configure_optimizers_cyclic_lr_interval_step(self):
        """Test that CyclicLR scheduler returns interval='step'."""
        scheduler_config = {
            "type": "CyclicLR",
            "max_lr": 1e-2,
            "step_size_up": 1000,
        }
        p = PyanNet(
            scheduler_config=scheduler_config, learning_rate=1e-3, dataloader_length=100
        )
        result = p.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"
        assert result["lr_scheduler"]["frequency"] == 1

    def test_configure_optimizers_cyclic_lr_epoch_conversion(self):
        """Test that step_size_up_epoch is correctly converted to step_size_up."""
        dataloader_length = 50
        step_size_up_epochs = 4
        expected_step_size_up = dataloader_length * step_size_up_epochs

        scheduler_config = {
            "type": "CyclicLR",
            "max_lr": 1e-2,
            "step_size_up_epoch": step_size_up_epochs,
            "mode": "triangular",
        }
        p = PyanNet(
            scheduler_config=scheduler_config,
            learning_rate=1e-3,
            dataloader_length=dataloader_length,
        )
        result = p.configure_optimizers()

        # Access the underlying scheduler

        scheduler = result["lr_scheduler"]["scheduler"]
        # CyclicLR stores total_size = step_size_up + step_size_down
        # By default, step_size_down = step_size_up, so total_size = 2 * step_size_up
        assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        assert scheduler.total_size == 2 * expected_step_size_up
        assert scheduler.base_lrs[0] == 1e-3
        assert scheduler.max_lrs[0] == 1e-2

    def test_configure_optimizers_cyclic_lr_missing_dataloader_length_error(self):
        """Test that error is raised when dataloader_length is missing."""
        scheduler_config = {
            "type": "CyclicLR",
            "max_lr": 1e-2,
            "step_size_up_epoch": 4,
        }
        p = PyanNet(
            scheduler_config=scheduler_config,
            learning_rate=1e-3,
            dataloader_length=None,
        )
        expected_msg = "step_size_up_epoch requires dataloader_length to be set."
        with pytest.raises(ValueError, match=expected_msg):
            p.configure_optimizers()

    def test_configure_optimizers_reduce_lr_on_plateau_interval_epoch(self):
        """Test that ReduceLROnPlateau scheduler returns interval='epoch'."""
        scheduler_config = {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "monitor": "val_loss",
        }
        p = PyanNet(scheduler_config=scheduler_config, learning_rate=1e-3)
        result = p.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "epoch"
        assert result["lr_scheduler"]["monitor"] == "val_loss"

    def test_configure_optimizers_cyclic_lr_with_step_size_up(self):
        """Test that step_size_up can be provided directly."""
        scheduler_config = {
            "type": "CyclicLR",
            "max_lr": 1e-2,
            "step_size_up": 1500,
            "mode": "triangular",
        }
        p = PyanNet(scheduler_config=scheduler_config, learning_rate=1e-3)
        result = p.configure_optimizers()

        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        # total_size = step_size_up + step_size_down = 2 * step_size_up (default)
        assert scheduler.total_size == 2 * 1500
        assert scheduler.base_lrs[0] == 1e-3
        assert scheduler.max_lrs[0] == 1e-2

    def test_configure_optimizers_cyclic_lr_default_with_warning(self):
        """
        Test that default step_size_up is used with warning when neither are provided.
        """
        scheduler_config = {
            "type": "CyclicLR",
            "max_lr": 1e-2,
            "mode": "triangular",
        }
        p = PyanNet(scheduler_config=scheduler_config, learning_rate=1e-3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.configure_optimizers()

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            msg = (
                "Neither 'step_size_up' nor 'step_size_up_epoch' specified in "
                "CyclicLR config. Using default step_size_up=2000"
            )
            assert str(w[0].message) == msg

        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        # Default step_size_up = 2000, so total_size = 4000
        assert scheduler.total_size == 2 * 2000
