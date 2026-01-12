from unittest.mock import Mock

import torch
import torch.nn.functional as F

from dr_sad.models.REx import VRExModel
from dr_sad.utils import TrainingBatch


class TestVRExLoss:
    """Test cases for VREx loss in VRExModel."""

    def test_vrex_loss_initialization(self):
        """Test VRExModel initialization."""
        lambda_vrex = 10.0
        model = VRExModel(lambda_vrex=lambda_vrex)
        assert model.lambda_vrex == lambda_vrex

    def test_vrex_loss_forward_single_domain(self):
        """Test VREx loss forward pass with a single domain."""
        model = VRExModel(lambda_vrex=5.0)
        batch_size, num_classes, num_frames = 3, 1, 4
        logits = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        domain_ids = torch.zeros(batch_size, dtype=torch.long)

        # Use BCE as model.loss_function
        loss, metrics = model.vrex_loss(logits, labels, domain_ids)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "erm_loss" in metrics
        assert "vrex_penalty" in metrics
        assert not metrics["erm_loss"].requires_grad
        assert not metrics["vrex_penalty"].requires_grad

    def test_vrex_loss_forward_multiple_domains(self):
        """Test VREx loss forward pass with multiple domains."""
        model = VRExModel(lambda_vrex=2.0)
        batch_size, num_classes, num_frames = 6, 1, 3
        logits = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        domain_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        loss, metrics = model.vrex_loss(logits, labels, domain_ids)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert metrics["erm_loss"].item() > 0
        assert metrics["vrex_penalty"].item() >= 0

    def test_vrex_loss_backward_pass(self):
        """Test that VREx loss supports gradient computation."""
        model = VRExModel(lambda_vrex=1.0)
        batch_size, num_classes, num_frames = 4, 1, 5
        logits = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        domain_ids = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        loss, _ = model.vrex_loss(logits, labels, domain_ids)
        loss.backward()
        assert logits.grad is not None

    def test_erm_loss_equals_cross_entropy_when_lambda_zero(self):
        """Test that ERM loss equals BCE when lambda_vrex=0."""
        model = VRExModel(lambda_vrex=0.0)
        batch_size, num_classes, num_frames = 3, 1, 6
        logits = torch.rand(batch_size, num_classes, num_frames, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        domain_ids = torch.zeros(batch_size, dtype=torch.long)
        vrex_loss, metrics = model.vrex_loss(logits, labels, domain_ids)
        expected_loss = F.binary_cross_entropy(logits, labels)
        assert torch.allclose(vrex_loss, expected_loss, atol=1e-6)
        assert torch.allclose(metrics["erm_loss"], expected_loss, atol=1e-6)

    def test_vrex_model_training_step(self):
        """Test VRExModel training_step method."""
        model = VRExModel(lambda_vrex=5.0)

        # Mock parent class methods
        model.prepare_annotation = Mock()  # type: ignore[method-assign]
        model.log = Mock()
        model.step_waterfall_scheduler = Mock()  # type: ignore[method-assign]

        # Create mock batch
        batch_size, channels, samples = 2, 1, 1600
        waveforms = torch.randn(batch_size, channels, samples)
        annotations = [[(0.1, 0.5), (0.7, 0.9)], [(0.2, 0.6)]]
        domains = torch.tensor([0, 1], dtype=torch.long)

        batch = TrainingBatch(
            waveforms=waveforms, annotations=annotations, domains=domains
        )

        num_frames = 10
        num_classes = 1
        # Model should output probabilities of shape (batch, frames, classes)
        model_outputs = torch.rand(
            batch_size, num_frames, num_classes, requires_grad=True
        )

        model.forward = Mock(return_value=model_outputs)  # type: ignore[method-assign]

        # Mock prepare_annotation output: (batch, 1, frames)
        speaker_truth = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        model.prepare_annotation.return_value = speaker_truth

        # Run training step
        loss = model.training_step(batch, batch_idx=0)

        # Verify calls
        model.prepare_annotation.assert_called_once_with(waveforms, annotations)
        model.forward.assert_called_once_with(waveforms)
        model.step_waterfall_scheduler.assert_not_called()  # No scheduling in this test

        # Check logging calls - log is called 4 times:
        # train_loss, train_erm, train_vrex_penalty, lambda_vrex
        assert model.log.call_count == 4

        # Verify loss is computed and has gradients
        assert isinstance(loss, torch.Tensor)
        # Check the model outputs still have grad
        assert model_outputs.requires_grad

    def test_vrex_model_training_step_with_scheduling(self):
        """Test VRExModel training_step with lambda scheduling."""
        model = VRExModel(lambda_vrex=10.0, lambda_scheduling_steps=1)

        # Verify initial state
        assert model.lambda_vrex == 0.0
        assert model.target_lambda == 10.0
        assert model.anneal_step == 0

        # Mock parent class methods
        model.prepare_annotation = Mock()  # type: ignore[method-assign]
        model.log = Mock()

        # Create mock batch
        batch_size, channels, samples = 2, 1, 1600
        waveforms = torch.randn(batch_size, channels, samples)
        annotations = [[(0.1, 0.5)], [(0.2, 0.6)]]
        domains = torch.tensor([0, 1], dtype=torch.long)

        batch = TrainingBatch(
            waveforms=waveforms, annotations=annotations, domains=domains
        )

        # Mock model forward pass
        num_frames = 8
        num_classes = 1
        model_outputs = torch.rand(batch_size, num_frames, num_classes)
        model.forward = Mock(return_value=model_outputs)  # type: ignore[method-assign]

        # Mock prepare_annotation output
        speaker_truth = torch.randint(0, 2, (batch_size, 1, num_frames)).float()
        model.prepare_annotation.return_value = speaker_truth

        # Run first training step - should keep lambda_vrex at 0.0
        loss1 = model.training_step(batch, batch_idx=0)
        assert model.anneal_step == 1
        assert model.lambda_vrex == 0.0
        assert isinstance(loss1, torch.Tensor)
        model.forward.assert_called_once_with(waveforms)

        # Reset mock call counts for second step
        model.log.reset_mock()
        model.forward.reset_mock()
        model.prepare_annotation.reset_mock()

        # Run second training step - should update lambda_vrex to target_lambda
        loss2 = model.training_step(batch, batch_idx=1)

        # After second step: lambda_vrex should now be target_lambda
        assert model.anneal_step == 2
        assert model.lambda_vrex == 10.0

        # Verify calls for second step
        model.prepare_annotation.assert_called_once_with(waveforms, annotations)
        model.forward.assert_called_once_with(waveforms)
        assert model.log.call_count == 4

        # Verify loss is computed
        assert isinstance(loss2, torch.Tensor)
