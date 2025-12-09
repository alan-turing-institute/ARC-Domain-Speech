from unittest.mock import Mock

import torch
import torch.nn.functional as F

from dr_sad.models.IRM import IRMLoss, IRMModel, TrainingBatch
from dr_sad.pyannet import PyanNet


class TestIRMLoss:
    """Test cases for IRMLoss class."""

    def test_irm_loss_initialization(self):
        """Test IRMLoss initialization with default and custom lambda values."""
        # Default lambda
        loss_fn = IRMLoss()
        assert loss_fn.lambda_irm == 1e2
        assert isinstance(loss_fn.dummy_w, torch.nn.Parameter)
        assert loss_fn.dummy_w.item() == 1.0

        # Custom lambda
        custom_lambda = 50.0
        loss_fn_custom = IRMLoss(lambda_irm=custom_lambda)
        assert loss_fn_custom.lambda_irm == custom_lambda

    def test_irm_loss_forward_single_environment(self):
        """Test IRMLoss forward pass with a single environment."""
        loss_fn = IRMLoss(lambda_irm=10.0)

        # Create test data: (batch=2, channels=1, frames=4) for binary classification
        batch_size, num_classes, num_frames = 2, 1, 4
        logits = torch.randn(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames))  # Binary: 0 or 1
        env_ids = torch.zeros(batch_size, dtype=torch.long)  # Single environment

        # Forward pass
        loss, metrics = loss_fn(logits, labels, env_ids)

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
        loss_fn = IRMLoss(lambda_irm=5.0)

        # Create test data: (batch=6, channels=1, frames=3) for binary classification
        batch_size, num_classes, num_frames = 6, 1, 3
        logits = torch.randn(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames))  # Binary: 0 or 1
        # Three environments with 2 samples each
        env_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        # Forward pass
        loss, metrics = loss_fn(logits, labels, env_ids)

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
        loss_fn = IRMLoss(lambda_irm=1.0)

        # Create test data for binary classification
        batch_size, num_classes, num_frames = 3, 1, 5
        logits = torch.randn(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames))  # Binary: 0 or 1
        env_ids = torch.tensor([0, 1, 1], dtype=torch.long)

        # Forward and backward pass
        loss, _ = loss_fn(logits, labels, env_ids)
        loss.backward()

        # Check gradients exist
        assert logits.grad is not None
        assert loss_fn.dummy_w.grad is not None

    def test_erm_loss_equals_cross_entropy_when_lambda_zero(self):
        """Test that ERM loss equals standard binary cross-entropy when lambda_irm=0."""
        loss_fn = IRMLoss(lambda_irm=0.0)

        # Create test data with single environment for binary classification
        batch_size, num_classes, num_frames = 4, 1, 5
        logits = torch.randn(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames))  # Binary: 0 or 1
        env_ids = torch.zeros(batch_size, dtype=torch.long)  # Single environment

        # Forward pass through IRMLoss
        irm_loss, metrics = loss_fn(logits, labels, env_ids)

        # Compute standard binary cross-entropy manually
        # IRMLoss transforms: logits.transpose(1, 2), labels.squeeze(1)
        logits_transformed = logits.transpose(1, 2)  # (batch, frames, 1)
        labels_transformed = labels.squeeze(1).float()  # (batch, frames)

        # For binary cross-entropy, squeeze logits to match labels
        logits_flat = logits_transformed.squeeze(-1)  # (batch, frames)

        expected_loss = F.binary_cross_entropy_with_logits(
            logits_flat, labels_transformed
        )

        # When lambda_irm=0 and single environment, total loss should equal standard BCE
        assert torch.allclose(irm_loss, expected_loss, atol=1e-6)
        assert torch.allclose(metrics["erm_loss"], expected_loss, atol=1e-6)


class TestIRMModel:
    """Test cases for IRMModel class."""

    def test_irm_model_initialization(self):
        """Test IRMModel initialization."""
        # Test default lambda
        model = IRMModel()
        assert hasattr(model, "irm_loss")
        assert isinstance(model.irm_loss, IRMLoss)
        assert model.irm_loss.lambda_irm == 1e2

        # Test custom lambda
        custom_lambda = 25.0
        model_custom = IRMModel(lambda_irm=custom_lambda)
        assert model_custom.irm_loss.lambda_irm == custom_lambda

    def test_irm_model_training_step(self):
        """Test IRMModel training_step method."""
        model = IRMModel(lambda_irm=10.0)

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
        # Model should output (batch, time, channels)
        model_outputs = torch.randn(batch_size, num_frames, num_classes)

        def mock_forward(_):
            return model_outputs

        model.forward = mock_forward  # type: ignore[method-assign]

        # Mock prepare_annotation output: (batch, 1, frames) with binary labels
        speaker_truth = torch.randint(
            0, 2, (batch_size, 1, num_frames)
        )  # Binary: 0 or 1
        model.prepare_annotation.return_value = speaker_truth

        # Run training step
        loss = model.training_step(batch, batch_idx=0)

        # Verify calls
        model.prepare_annotation.assert_called_once_with(waveforms, annotations)

        # Check logging calls - just verify that log was called 3 times
        assert model.log.call_count == 3

        # Verify loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_irm_model_only_overrides_training_step(self):
        """Test that IRMModel only overrides training_step, not val/test steps."""
        model = IRMModel(lambda_irm=10.0)
        parent_class = PyanNet  # PyanNet

        # Check that training_step is overridden
        assert model.__class__.training_step != parent_class.training_step

        # Check that validation_step and test_step are NOT overridden
        assert (
            not hasattr(model.__class__, "validation_step")
            or model.__class__.validation_step == parent_class.validation_step
        )
        assert (
            not hasattr(model.__class__, "test_step")
            or model.__class__.test_step == parent_class.test_step
        )
