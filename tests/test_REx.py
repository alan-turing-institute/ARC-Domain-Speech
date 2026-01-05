import torch
import torch.nn.functional as F

from dr_sad.models.REx import VRExModel


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
