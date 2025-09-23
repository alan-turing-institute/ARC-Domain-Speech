"""This file contains tests for the SincNet feature extractor."""

import pytest
import torch

from dr_sad.pyannet.sincnet import Abs, SincNet


class TestAbs:
    """Test the Abs layer."""

    def test_abs_forward(self):
        """Test that the Abs layer computes absolute values correctly."""
        abs_layer = Abs()

        # Test with positive values
        positive_input = torch.tensor([1.0, 2.0, 3.0])
        output = abs_layer(positive_input)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(output, expected)

        # Test with negative values
        negative_input = torch.tensor([-1.0, -2.0, -3.0])
        output = abs_layer(negative_input)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(output, expected)

        # Test with mixed values
        mixed_input = torch.tensor([-1.0, 2.0, -3.0, 4.0])
        output = abs_layer(mixed_input)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(output, expected)


class TestSincNet:
    """Test the SincNet feature extractor."""

    def test_init_default_params(self):
        """Test SincNet initialization with default parameters."""
        sincnet = SincNet()
        assert sincnet.sample_rate == 16000
        assert sincnet.stride == 10

        # Check that all blocks are properly initialized
        assert isinstance(sincnet.block0, torch.nn.Sequential)
        assert isinstance(sincnet.block1, torch.nn.Sequential)
        assert isinstance(sincnet.block2, torch.nn.Sequential)
        assert isinstance(sincnet.features, torch.nn.Sequential)

        # Check block lengths
        assert len(sincnet.block0) == 6  # norm, encoder, abs, pool, norm, lrelu
        assert len(sincnet.block1) == 4  # conv, pool, norm, lrelu
        assert len(sincnet.block2) == 4  # conv, pool, norm, lrelu
        assert len(sincnet.features) == 3  # block0, block1, block2

    def test_init_custom_params(self):
        """Test SincNet initialization with custom parameters."""
        sample_rate = 16000
        stride = 5
        sincnet = SincNet(sample_rate=sample_rate, stride=stride)
        assert sincnet.sample_rate == sample_rate
        assert sincnet.stride == stride

    def test_out_features(self):
        """Test that SincNet has the correct number of output features."""
        sincnet = SincNet()
        assert sincnet.out_features == 60

    def test_init_unsupported_sample_rate(self):
        """Test that unsupported sample rates raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="SincNet only supports 16kHz audio"
        ):
            SincNet(sample_rate=8000)

        with pytest.raises(
            NotImplementedError, match="SincNet only supports 16kHz audio"
        ):
            SincNet(sample_rate=44100)

    def test_forward_shape(self):
        """Test that forward pass produces expected output shapes."""
        sincnet = SincNet()

        # Test with different input lengths - use larger inputs to avoid size issues
        batch_size = 2

        for num_samples in [3200, 6400, 16000]:  # 0.2s, 0.4s, 1s at 16kHz
            input_tensor = torch.randn(batch_size, 1, num_samples)
            output = sincnet(input_tensor)

            # Check output shape
            assert output.dim() == 3  # (batch, channels, time)
            assert output.shape[0] == batch_size  # batch dimension preserved
            assert output.shape[1] == 60  # 60 output channels

            # Check that output length is reasonable
            expected_frames = sincnet.num_frames(num_samples)
            assert output.shape[2] == expected_frames

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        sincnet = SincNet()
        sincnet.eval()  # Set to eval mode to ensure deterministic behavior

        # Create input - use larger size to avoid conv size issues
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 1, 3200)

        # Run forward pass twice
        output1 = sincnet(input_tensor)
        output2 = sincnet(input_tensor)

        # Outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_num_frames_calculation(self):
        """Test num_frames method against empirically determined values."""
        sincnet = SincNet(stride=10)

        # Test with specific input lengths - these are empirically determined
        test_cases = [
            (3200, 9),  # Medium input
            (6400, 21),  # Larger input
            (16000, 56),  # 1 second input
        ]

        for num_samples, expected_frames in test_cases:
            calculated_frames = sincnet.num_frames(num_samples)
            # Use approximate comparison since exact values may vary
            assert calculated_frames == expected_frames, (
                f"Expected {expected_frames} frames for "
                f"{num_samples} samples, got {calculated_frames}"
            )

    def test_num_frames_consistency_with_forward(self):
        """Test that num_frames calculation matches actual forward pass output."""
        sincnet = SincNet()

        for num_samples in [3200, 6400, 8000, 16000]:
            input_tensor = torch.randn(1, 1, num_samples)
            output = sincnet(input_tensor)

            expected_frames = sincnet.num_frames(num_samples)
            actual_frames = output.shape[2]

            assert actual_frames == expected_frames, (
                f"num_frames({num_samples}) = {expected_frames}, "
                f"but forward pass produced {actual_frames} frames"
            )

    def test_receptive_field_size(self):
        """Test receptive field size calculation."""
        sincnet = SincNet(stride=10)

        # Test with different numbers of frames - use empirical values
        test_cases = [
            (1, 991),  # Single frame
            (5, 2071),  # Multiple frames
            (10, 3421),  # More frames
        ]

        for num_frames, expected_size in test_cases:
            calculated_size = sincnet.receptive_field_size(num_frames)
            # Allow some tolerance for calculation differences
            assert calculated_size == expected_size, (
                f"Expected receptive field size {expected_size} for "
                f"{num_frames} frames, got {calculated_size}"
            )

    def test_receptive_field_center(self):
        """Test receptive field center calculation."""
        sincnet = SincNet(stride=10)

        # Test with different frame indices - use empirical values
        test_cases = [
            (0, 495),  # First frame
            (1, 765),  # Second frame
            (5, 1845),  # Fifth frame
        ]

        for frame, expected_center in test_cases:
            calculated_center = sincnet.receptive_field_center(frame)
            # Allow some tolerance for calculation differences
            assert calculated_center == expected_center, (
                f"Expected receptive field center {expected_center} for "
                f"frame {frame}, got {calculated_center}"
            )

    def test_different_strides(self):
        """Test SincNet with different stride values."""
        for stride in [1, 5, 10, 20]:
            sincnet = SincNet(stride=stride)

            # Test forward pass with larger input to avoid size issues
            input_tensor = torch.randn(2, 1, 3200)
            output = sincnet(input_tensor)

            # Check basic properties
            assert output.shape[0] == 2  # batch size preserved
            assert output.shape[1] == 60  # 60 output channels
            assert output.shape[2] > 0  # non-zero time dimension

            # Test that num_frames is consistent
            expected_frames = sincnet.num_frames(3200)
            assert output.shape[2] == expected_frames

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        sincnet = SincNet()
        sincnet.train()

        # Create input with gradient tracking - use larger size
        input_tensor = torch.randn(1, 1, 3200, requires_grad=True)

        # Forward pass
        output = sincnet(input_tensor)

        # Create a simple loss (sum of outputs)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that input gradients exist and are non-zero
        assert input_tensor.grad is not None
        assert not torch.allclose(
            input_tensor.grad, torch.zeros_like(input_tensor.grad)
        )

    def test_batch_processing(self):
        """Test that the model can handle different batch sizes."""
        sincnet = SincNet()
        num_samples = 3200  # Use larger size to avoid conv issues

        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, 1, num_samples)
            output = sincnet(input_tensor)

            assert output.shape[0] == batch_size
            assert output.shape[1] == 60
            assert output.shape[2] == sincnet.num_frames(num_samples)
