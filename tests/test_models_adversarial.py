import numpy as np
import torch

from dr_sad.models.adversarial import AdversarialNet, GradientReversalLayer


class TestGradientReversalLayer:
    def test_forward_and_backward(self):
        """Test forward and backward pass of GradientReversalLayer."""
        grl = GradientReversalLayer(lambda_=1.5)
        x = torch.tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)

        # Forward pass should be identity
        y = grl(x)
        assert torch.allclose(y, x)

        # Backward pass should scale gradients by -lambda_
        y.sum().backward()
        expected_grad = torch.tensor([[-1.5, -1.5], [-1.5, -1.5]])
        assert torch.allclose(x.grad, expected_grad)


class TestAdversarialNet:
    def test_instantiation_and_basic_props(self):
        """Test basic instantiation and properties of AdversarialNet."""
        model = AdversarialNet(num_domains=5, grl_lambda=0.5)

        # basic properties inherited from PyanNet
        assert isinstance(model.sample_rate, int)
        assert model.sample_rate == 16000
        assert model.num_domains == 5
        assert model.domain_loss_weight == 1.0

        num_frames = model.num_frames(16000)
        assert isinstance(num_frames, int)
        assert num_frames > 0

        rf_size = model.receptive_field_size(1)
        assert isinstance(rf_size, int)
        assert rf_size > 0

        center = model.receptive_field_center(0)
        assert isinstance(center, int)

    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom domain parameters."""
        model = AdversarialNet(num_domains=3, domain_loss_weight=0.5, grl_lambda=0.8)

        assert model.num_domains == 3
        assert model.domain_loss_weight == 0.5
        assert model.domain_grl.lambda_ == 0.8

    def test_forward_and_output_shapes(self):
        """Test forward pass and output shapes."""
        model = AdversarialNet(num_domains=3, grl_lambda=0.5)
        batch = 2
        samples = 16000
        x = torch.randn(batch, 1, samples)

        # run forward on CPU without grad
        with torch.no_grad():
            speaker_scores, domain_logits = model(x)

        # Check speaker scores shape (batch, frames, classes)
        assert speaker_scores.ndim == 3
        assert speaker_scores.shape[0] == batch
        assert speaker_scores.shape[2] == model.num_classes

        # Check domain logits shape (batch, frames, num_domains)
        assert domain_logits.ndim == 3
        assert domain_logits.shape[0] == batch
        assert domain_logits.shape[1] == speaker_scores.shape[1]  # same num frames
        assert domain_logits.shape[2] == 3  # num_domains

        # Speaker scores should be in [0, 1] due to sigmoid activation
        assert torch.all(speaker_scores >= 0.0)
        assert torch.all(speaker_scores <= 1.0)

    def test_frame_centers_and_start_step(self):
        """Test frame center calculations (inherited from PyanNet)."""
        model = AdversarialNet(num_domains=2, grl_lambda=0.5)
        start, step = model.frame_centers_start_step()
        assert isinstance(start, float)
        assert isinstance(step, float)

        centers = model.frame_centers(16000, as_numpy=True)
        assert isinstance(centers, np.ndarray)
        assert centers.shape[0] == model.num_frames(16000)
        # centers should be strictly increasing
        assert np.all(np.diff(centers) > 0)

    def test_prepare_annotation_shape(self):
        """Test annotation preparation (inherited from PyanNet)."""
        model = AdversarialNet(num_domains=2, grl_lambda=0.5)
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = model.prepare_annotation(waveforms, ann)

        assert isinstance(speaker_truth, torch.Tensor)
        # (batch, 1, frames)
        assert speaker_truth.shape[0] == 1
        assert speaker_truth.shape[1] == 1
        assert speaker_truth.shape[2] == model.num_frames(waveforms.shape[-1])
        assert speaker_truth[0, 0, 2] == 1.0
        assert speaker_truth[0, 0, -2] == 0.0

    def test_loss_output_function(self):
        """Test speaker loss function."""
        model = AdversarialNet(num_domains=3, grl_lambda=0.5)
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = model.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)

        loss = model.loss_output_function(speaker_truth, [0], outputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_loss_domain_function(self):
        """Test domain loss function."""
        model = AdversarialNet(num_domains=3, grl_lambda=0.5)
        batch_size = 2
        num_frames = 100

        # Create domain logits: (batch, frames, num_domains)
        domain_logits = torch.randn(batch_size, num_frames, 3)
        domains = [0, 2]  # domain indices for each sample in batch
        speaker_truth = torch.zeros(batch_size, 1, num_frames)  # dummy, not used

        loss = model.loss_domain_function(speaker_truth, domains, domain_logits)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_combined_loss_function(self):
        """Test combined adversarial loss function."""
        model = AdversarialNet(num_domains=3, domain_loss_weight=0.5, grl_lambda=0.5)
        waveforms = torch.zeros(2, 1, 16000)
        ann = [[(0.0, 0.1)], [(0.5, 1.0)]]
        domains = [0, 1]

        speaker_truth = model.prepare_annotation(waveforms, ann)
        speaker_outputs = torch.rand_like(speaker_truth)
        domain_logits = torch.randn(2, speaker_truth.shape[2], 3)

        # Test individual loss components
        total_loss, speaker_loss, domain_loss = model._compute_losses(
            speaker_truth, domains, speaker_outputs, domain_logits
        )

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(speaker_loss, torch.Tensor)
        assert isinstance(domain_loss, torch.Tensor)
        assert total_loss.ndim == 0

        # Test combined loss function
        combined_loss = model.loss_function(
            speaker_truth, domains, speaker_outputs, domain_logits
        )
        assert torch.isclose(combined_loss, total_loss)

    def test_domain_accuracy_function(self):
        """Test domain accuracy calculation."""
        model = AdversarialNet(num_domains=3, grl_lambda=0.5)
        batch_size = 4
        num_frames = 50

        # Create domain logits where predictions should be correct
        domain_logits = torch.zeros(batch_size, num_frames, 3)
        domains = [0, 1, 2, 0]

        # Make predictions clearly favor the correct domains
        for i, domain in enumerate(domains):
            domain_logits[i, :, domain] = 5.0  # high score for correct domain

        accuracy = model.domain_accuracy_function(domains, domain_logits)
        assert isinstance(accuracy, float)
        assert accuracy == 1.0  # should be perfect accuracy

        # Test with random logits
        random_logits = torch.randn(batch_size, num_frames, 3)
        random_accuracy = model.domain_accuracy_function(domains, random_logits)
        assert isinstance(random_accuracy, float)
        assert 0.0 <= random_accuracy <= 1.0

    def test_accuracy_function(self):
        """Test speaker accuracy function (inherited from PyanNet)."""
        model = AdversarialNet(num_domains=2, grl_lambda=0.5)
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = model.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)

        acc = model.accuracy_function(speaker_truth, [0], outputs, threshold=0.5)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_predict_step(self):
        """Test predict step functionality."""
        model = AdversarialNet(num_domains=2, grl_lambda=0.5)
        batch = {"waveforms": torch.randn(1, 1, 16000)}

        with torch.no_grad():
            outputs = model.predict_step(batch, 0)

        # Should return only speaker outputs, swapped to (batch, channels, time)
        assert isinstance(outputs, torch.Tensor)
        assert outputs.ndim == 3
        assert outputs.shape[0] == 1
        assert outputs.shape[1] == model.num_classes

    def test_evaluate_batch(self):
        """Test evaluate_batch functionality."""
        model = AdversarialNet(num_domains=2, grl_lambda=0.5)
        waveforms = torch.randn(2, 1, 16000)
        annotations = [[(0.0, 0.5)], [(1.0, 1.5)]]
        domains = [0, 1]

        batch = {"waveforms": waveforms, "annotations": annotations, "domains": domains}

        with torch.no_grad():
            result = model.evaluate_batch(batch, 0)

        total_loss, accuracy, speaker_loss, domain_loss, domain_accuracy = result

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(accuracy, float)
        assert isinstance(speaker_loss, torch.Tensor)
        assert isinstance(domain_loss, torch.Tensor)
        assert isinstance(domain_accuracy, float)

        assert total_loss.ndim == 0
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= domain_accuracy <= 1.0

    def test_domain_components_exist(self):
        """Test that domain-specific components are properly initialized."""
        model = AdversarialNet(num_domains=4, grl_lambda=0.5)

        # Check that domain components exist
        assert hasattr(model, "domain_grl")
        assert hasattr(model, "domain_linear")
        assert hasattr(model, "domain_classifier")

        # Check their properties
        assert model.domain_grl.lambda_ == 0.5  # default
        assert model.domain_linear.out_features > 0
        assert model.domain_classifier.out_features == 4  # num_domains
