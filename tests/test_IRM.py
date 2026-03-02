from unittest.mock import Mock

import torch
import torch.nn.functional as F

from dr_sad.models.IRM import IRMv1Model
from dr_sad.utils import TrainingBatch


class TestIRMLoss:
    """Test cases for IRMLoss class."""

    def test_irm_loss_initialization(self):
        """Test IRMLoss initialization."""
        custom_lambda = 50.0
        lambda_scheduling_epochs = 2
        dataloader_length = 100
        model = IRMv1Model(
            lambda_irm=custom_lambda,
            lambda_scheduling_epochs=lambda_scheduling_epochs,
            dataloader_length=dataloader_length,
        )
        assert model.lambda_irm == 0.0  # Starts at 0 with scheduling
        assert (
            model.lambda_scheduling_steps
            == lambda_scheduling_epochs * dataloader_length
        )
        assert model.target_lambda == custom_lambda

        # test without scheduling
        model_no_schedule = IRMv1Model(
            lambda_irm=custom_lambda,
            lambda_scheduling_epochs=None,
        )
        assert model_no_schedule.lambda_irm == custom_lambda
        assert model_no_schedule.target_lambda == custom_lambda
        assert model_no_schedule.lambda_scheduling_steps is None

    def test_irm_loss_forward_single_environment(self):
        """Test IRMLoss forward pass with a single environment."""
        model = IRMv1Model(lambda_irm=10.0, lambda_scheduling_epochs=None)

        # Create test data: (batch=2, channels=1, frames=4) for binary classification
        batch_size, num_classes, num_frames = 2, 1, 4
        # Create probabilities directly (as would come from model forward pass)
        probs = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        ).float()  # Binary: 0 or 1, convert to float
        env_ids = torch.zeros(batch_size, dtype=torch.long)  # Single environment

        # Forward pass
        loss, metrics = model.irm_loss(probs, labels, env_ids)

        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

        assert "erm_loss" in metrics
        assert "irm_penalty" in metrics
        assert isinstance(metrics["erm_loss"], torch.Tensor)
        assert isinstance(metrics["irm_penalty"], torch.Tensor)
        assert not metrics["erm_loss"].requires_grad  # Should be detached
        assert not metrics["irm_penalty"].requires_grad  # Should be detached

    def test_irm_loss_forward_multiple_environments(self):
        """Test IRMLoss forward pass with multiple environments."""
        model = IRMv1Model(lambda_irm=5.0, lambda_scheduling_epochs=None)

        # Create test data: (batch=6, channels=1, frames=3) for binary classification
        batch_size, num_classes, num_frames = 6, 1, 3
        # Create probabilities directly (as would come from model forward pass)
        probs = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        ).float()  # Binary: 0 or 1, convert to float
        # Three environments with 2 samples each
        env_ids = torch.tensor([0, 0, 1, 1, 2, 2])

        # Forward pass
        loss, metrics = model.irm_loss(probs, labels, env_ids)

        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

        # With multiple environments, both ERM and penalty should be non-zero
        assert metrics["erm_loss"].item() > 0
        # Penalty can be zero if gradients are zero
        assert metrics["irm_penalty"].item() >= 0

    def test_irm_loss_backward_pass(self):
        """Test that IRMLoss supports gradient computation."""
        model = IRMv1Model(lambda_irm=1.0, lambda_scheduling_epochs=None)

        # Create test data for binary classification
        batch_size, num_classes, num_frames = 3, 1, 5
        # Create probabilities directly (as would come from model forward pass)
        probs = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        ).float()  # Binary: 0 or 1, convert to float
        env_ids = torch.tensor([0, 1, 1], dtype=torch.long)

        # Forward and backward pass
        loss, _ = model.irm_loss(probs, labels, env_ids)
        loss.backward()

        # Check gradients exist
        assert probs.grad is not None
        assert model.dummy_w.grad is not None

    def test_erm_loss_equals_cross_entropy_when_lambda_zero(self):
        """Test that ERM loss equals standard binary cross-entropy when lambda_irm=0."""
        model = IRMv1Model(lambda_irm=0.0, lambda_scheduling_epochs=None)

        # Create test data with single environment for binary classification
        batch_size, num_classes, num_frames = 4, 1, 5
        # Create probabilities directly (as would come from model forward pass)
        probs = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        ).float()  # Binary: 0 or 1, convert to float
        env_ids = torch.zeros(batch_size, dtype=torch.long)  # Single environment

        # Forward pass through IRMLoss
        irm_loss, metrics = model.irm_loss(probs, labels, env_ids)

        # Compute standard binary cross-entropy manually using same format
        # IRMLoss transforms: probs.transpose(1, 2), labels.squeeze(1)
        probs_transformed = probs.transpose(1, 2)  # (batch, frames, 1)
        labels_transformed = labels.squeeze(1).float()  # (batch, frames)

        # For binary cross-entropy with probabilities, squeeze probs to match labels
        probs_flat = probs_transformed.squeeze(-1)  # (batch, frames)
        expected_loss = F.binary_cross_entropy(probs_flat, labels_transformed)

        # When lambda_irm=0 and single environment, total loss should equal standard BCE
        assert torch.allclose(irm_loss, expected_loss, atol=1e-6)
        assert torch.allclose(metrics["erm_loss"], expected_loss, atol=1e-6)


class TestIRMModel:
    """Test cases for IRMModel class."""

    def test_irm_model_initialization(self):
        """Test IRMModel initialization."""
        # Test with explicit lambda (no default)
        model = IRMv1Model(lambda_irm=1e2, lambda_scheduling_epochs=None)
        # IRMLoss is now a method, not a class instance
        # Check that IRMLoss is callable
        assert callable(model.irm_loss)
        assert model.lambda_irm == 1e2

        # Test custom lambda
        custom_lambda = 25.0
        model_custom = IRMv1Model(lambda_irm=custom_lambda)
        assert model_custom.lambda_irm == custom_lambda

    def test_irm_model_training_step(self):
        """Test IRMModel training_step method."""
        model = IRMv1Model(lambda_irm=10.0)

        # Mock parent class methods
        model.prepare_annotation = Mock()  # type: ignore[method-assign]
        model.log = Mock()

        # Create mock batch
        batch_size, channels, samples = 2, 1, 1600
        waveforms = torch.randn(batch_size, channels, samples)
        annotations = [[(0.1, 0.5), (0.7, 0.9)], [(0.2, 0.6)]]
        domains = torch.tensor([0, 1], dtype=torch.long)

        batch = TrainingBatch(
            waveforms=waveforms, annotations=annotations, domains=domains
        )

        # Mock model forward pass properly by overriding the forward method
        num_frames = 10
        num_classes = 1  # Binary classification
        # Model should output (batch, time, channels) with probabilities (not logits)
        model_outputs = torch.rand(batch_size, num_frames, num_classes)

        def mock_forward(_):
            return model_outputs

        model.forward = mock_forward  # type: ignore[method-assign]

        # Mock prepare_annotation output: (batch, 1, frames) with binary labels
        speaker_truth = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        ).float()  # Binary: 0 or 1, convert to float
        model.prepare_annotation.return_value = speaker_truth

        # Run training step
        loss = model.training_step(batch, batch_idx=0)

        # Verify calls
        model.prepare_annotation.assert_called_once_with(waveforms, annotations)

        # Check logging calls - log is called 4 times: train_loss, train_erm,
        # train_penalty, lambda_irm
        assert model.log.call_count == 4

        # Verify loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
