"""Tests for dr_sad.pyannet.receptive_field module.

These tests are super excessive but we are in a brave new world. :robot:
"""

import numpy as np
import pytest

from dr_sad.pyannet import receptive_field


class TestConv1dNumFrames:
    """Tests for conv1d_num_frames function."""

    def test_basic_convolution(self):
        """Test basic convolution without padding or dilation."""
        # Standard convolution: output = (input - kernel + 1) / stride
        result = receptive_field.conv1d_num_frames(100, kernel_size=5, stride=1)
        expected = 96  # (100 - 5 + 1) = 96
        assert result == expected

    def test_convolution_with_stride(self):
        """Test convolution with stride > 1."""
        result = receptive_field.conv1d_num_frames(100, kernel_size=5, stride=2)
        expected = 48  # (100 - 5 + 1) / 2 = 48
        assert result == expected

    def test_convolution_with_padding(self):
        """Test convolution with padding."""
        result = receptive_field.conv1d_num_frames(
            100, kernel_size=5, stride=1, padding=2
        )
        expected = 100  # (100 + 2*2 - 5 + 1) = 100
        assert result == expected

    def test_convolution_with_dilation(self):
        """Test convolution with dilation."""
        result = receptive_field.conv1d_num_frames(
            100, kernel_size=3, stride=1, dilation=2
        )
        # effective_kernel = 1 + (3-1)*2 = 5, so (100 - 5 + 1) = 96
        expected = 96
        assert result == expected

    def test_convolution_all_parameters(self):
        """Test convolution with all parameters set."""
        result = receptive_field.conv1d_num_frames(
            num_samples=100, kernel_size=3, stride=2, padding=1, dilation=2
        )
        # effective_kernel = 1 + (3-1)*2 = 5
        # output = (100 + 2*1 - 5 + 1) / 2 = 98 / 2 = 49
        expected = 49
        assert result == expected

    def test_edge_case_minimal_input(self):
        """Test with minimal valid input."""
        result = receptive_field.conv1d_num_frames(1, kernel_size=1, stride=1)
        expected = 1
        assert result == expected

    def test_invalid_input_types(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="num_samples: expected int, got"):
            receptive_field.conv1d_num_frames(100.5, kernel_size=5)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="kernel_size: expected int, got"):
            receptive_field.conv1d_num_frames(100, kernel_size=5.5)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="stride: expected int, got"):
            receptive_field.conv1d_num_frames(100, kernel_size=5, stride="1")  # type: ignore[arg-type]


class TestMultiConvNumFrames:
    """Tests for multi_conv_num_frames function."""

    def test_single_layer(self):
        """Test with single convolution layer."""
        result = receptive_field.multi_conv_num_frames(
            num_samples=100,
            kernel_size=[5],
            stride=[1],
            padding=[0],
            dilation=[1],
        )
        expected = 96  # Same as conv1d_num_frames(100, 5, 1, 0, 1)
        assert result == expected

    def test_multiple_layers(self):
        """Test with multiple convolution layers."""
        result = receptive_field.multi_conv_num_frames(
            num_samples=100,
            kernel_size=[5, 3],
            stride=[1, 2],
            padding=[0, 1],
            dilation=[1, 1],
        )
        # First layer: (100 - 5 + 1) = 96
        # Second layer: (96 + 2*1 - 3 + 1) / 2 = 96 / 2 = 48
        expected = 48
        assert result == expected

    def test_three_layers(self):
        """Test with three convolution layers."""
        result = receptive_field.multi_conv_num_frames(
            num_samples=1000,
            kernel_size=[10, 5, 3],
            stride=[5, 2, 1],
            padding=[0, 1, 1],
            dilation=[1, 1, 2],
        )
        # Layer 1: (1000 + 2*0 - 1*(10-1) - 1) // 5 = 990 // 5 = 198
        # Actually: (1000 + 2*0 - 1*(10-1) - 1) // 5 + 1 = 199
        # Layer 2: (199 + 2*1 - 1*(5-1) - 1) // 2 + 1 = 99
        # Layer 3: effective_kernel = 1 + (3-1)*2 = 5
        # (99 + 2*1 - 5 - 1) // 1 + 1 = 97
        expected = 97
        assert result == expected

    def test_mismatched_lengths(self):
        """Test that mismatched parameter lengths raise ValueError."""
        msg = "kernel_size, stride, padding, dilation must have equal length"
        with pytest.raises(ValueError, match=msg):
            receptive_field.multi_conv_num_frames(
                100,
                kernel_size=[5, 3],
                stride=[1],
                padding=[0, 1],
                dilation=[1, 1],
            )


class TestConv1dReceptiveFieldSize:
    """Tests for conv1d_receptive_field_size function."""

    def test_basic_receptive_field(self):
        """Test basic receptive field calculation."""
        result = receptive_field.conv1d_receptive_field_size(
            num_frames=1, kernel_size=5, stride=1, padding=0, dilation=1
        )
        expected = 5  # effective_kernel=5, (5 + (1-1)*1 - 2*0) = 5
        assert result == expected

    def test_multiple_frames(self):
        """Test receptive field with multiple frames."""
        result = receptive_field.conv1d_receptive_field_size(
            num_frames=3, kernel_size=5, stride=2, padding=0, dilation=1
        )
        expected = 9  # effective_kernel=5, (5 + (3-1)*2 - 2*0) = 9
        assert result == expected

    def test_with_dilation(self):
        """Test receptive field with dilation."""
        result = receptive_field.conv1d_receptive_field_size(
            num_frames=1, kernel_size=3, stride=1, padding=0, dilation=3
        )
        # effective_kernel = 1 + (3-1)*3 = 7, (7 + (1-1)*1 - 2*0) = 7
        expected = 7
        assert result == expected

    def test_with_padding(self):
        """Test receptive field with padding."""
        result = receptive_field.conv1d_receptive_field_size(
            num_frames=1, kernel_size=5, stride=1, padding=2, dilation=1
        )
        expected = 1  # effective_kernel=5, (5 + (1-1)*1 - 2*2) = 1
        assert result == expected

    def test_invalid_input_types(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="num_frames: expected int, got"):
            receptive_field.conv1d_receptive_field_size(1.5, kernel_size=5)  # type: ignore[arg-type]


class TestMultiConvReceptiveFieldSize:
    """Tests for multi_conv_receptive_field_size function."""

    def test_single_layer(self):
        """Test with single layer."""
        result = receptive_field.multi_conv_receptive_field_size(
            num_frames=1,
            kernel_size=[5],
            stride=[1],
            padding=[0],
            dilation=[1],
        )
        expected = 5
        assert result == expected

    def test_multiple_layers(self):
        """Test with multiple layers."""
        result = receptive_field.multi_conv_receptive_field_size(
            num_frames=2,
            kernel_size=[3, 5],
            stride=[1, 2],
            padding=[0, 1],
            dilation=[1, 1],
        )
        # Working backwards:
        # Layer 2: effective_kernel=5, (5 + (2-1)*2 - 2*1) = 5
        # Layer 1: effective_kernel=3, (3 + (5-1)*1 - 2*0) = 7
        expected = 7
        assert result == expected


class TestConv1dReceptiveFieldCenter:
    """Tests for conv1d_receptive_field_center function."""

    def test_basic_center(self):
        """Test basic receptive field center calculation."""
        result = receptive_field.conv1d_receptive_field_center(
            frame=0, kernel_size=5, stride=1, padding=0, dilation=1
        )
        expected = 2  # 0*1 + (5-1)//2 - 0 = 2
        assert result == expected

    def test_center_with_stride(self):
        """Test receptive field center with stride."""
        result = receptive_field.conv1d_receptive_field_center(
            frame=2, kernel_size=3, stride=2, padding=0, dilation=1
        )
        expected = 5  # 2*2 + (3-1)//2 - 0 = 5
        assert result == expected

    def test_center_with_dilation(self):
        """Test receptive field center with dilation."""
        result = receptive_field.conv1d_receptive_field_center(
            frame=1, kernel_size=3, stride=1, padding=0, dilation=2
        )
        # effective_kernel = 1+(3-1)*2=5, 1*1 + (5-1)//2 - 0 = 3
        expected = 3
        assert result == expected

    def test_center_with_padding(self):
        """Test receptive field center with padding."""
        result = receptive_field.conv1d_receptive_field_center(
            frame=1, kernel_size=5, stride=1, padding=2, dilation=1
        )
        expected = 1  # 1*1 + (5-1)//2 - 2 = 1
        assert result == expected

    def test_different_frames(self):
        """Test receptive field center for different frame indices."""
        for frame in range(5):
            result = receptive_field.conv1d_receptive_field_center(
                frame=frame, kernel_size=3, stride=2, padding=0, dilation=1
            )
            expected = frame * 2 + 1  # frame*2 + (3-1)//2 - 0
            assert result == expected

    def test_invalid_input_types(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="frame: expected int, got"):
            receptive_field.conv1d_receptive_field_center(1.5, kernel_size=5)  # type: ignore[arg-type]


class TestMultiConvReceptiveFieldCenter:
    """Tests for multi_conv_receptive_field_center function."""

    def test_single_layer(self):
        """Test with single layer."""
        result = receptive_field.multi_conv_receptive_field_center(
            frame=1,
            kernel_size=[5],
            stride=[1],
            padding=[0],
            dilation=[1],
        )
        expected = 3  # 1*1 + (5-1)//2 - 0 = 3
        assert result == expected

    def test_multiple_layers(self):
        """Test with multiple layers."""
        result = receptive_field.multi_conv_receptive_field_center(
            frame=2,
            kernel_size=[3, 5],
            stride=[2, 1],
            padding=[0, 1],
            dilation=[1, 1],
        )
        # Working backwards:
        # Layer 2: 2*1 + (5-1)//2 - 1 = 3
        # Layer 1: 3*2 + (3-1)//2 - 0 = 7
        expected = 7
        assert result == expected

    def test_three_layers(self):
        """Test with three layers."""
        result = receptive_field.multi_conv_receptive_field_center(
            frame=1,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )
        # Working backwards through layers:
        # Layer 3: 1*1 + (3-1)//2 - 0 = 2
        # Layer 2: 2*1 + (3-1)//2 - 0 = 3
        # Layer 1: 3*1 + (3-1)//2 - 0 = 4
        expected = 4
        assert result == expected


class TestPrivateHelperFunctions:
    """Tests for private helper functions."""

    def test_check_int_valid(self):
        """Test _check_int with valid integers."""
        # Should not raise any exception
        receptive_field._check_int("test", 5)
        receptive_field._check_int("test", 0)
        receptive_field._check_int("test", -1)

    def test_check_int_invalid(self):
        """Test _check_int with invalid types."""
        with pytest.raises(TypeError, match="test: expected int, got"):
            receptive_field._check_int("test", 5.5)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="test: expected int, got"):
            receptive_field._check_int("test", "5")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="test: expected int, got"):
            receptive_field._check_int("test", None)  # type: ignore[arg-type]

    def test_check_lists_valid(self):
        """Test _check_lists with valid lists."""
        # Should not raise any exception
        receptive_field._check_lists([1, 2], [1, 2], [0, 0], [1, 1])
        receptive_field._check_lists([5], [1], [0], [1])

    def test_check_lists_none_values(self):
        """Test _check_lists with None values."""
        msg = "kernel_size, stride, padding, dilation must all be provided"
        with pytest.raises(ValueError, match=msg):
            receptive_field._check_lists(None, [1], [0], [1])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=msg):
            receptive_field._check_lists([1], None, [0], [1])  # type: ignore[arg-type]

    def test_check_lists_mismatched_lengths(self):
        """Test _check_lists with mismatched lengths."""
        msg = "kernel_size, stride, padding, dilation must have equal length"
        with pytest.raises(ValueError, match=msg):
            receptive_field._check_lists([1, 2], [1], [0, 0], [1, 1])

        with pytest.raises(ValueError, match=msg):
            receptive_field._check_lists([1], [1, 2], [0], [1])


class TestRealisticScenarios:
    """Test realistic neural network scenarios."""

    def test_sincnet_like_architecture(self):
        """Test scenario similar to SincNet architecture from the codebase."""
        # Based on the SincNet code: stride=10 in first layer, then stride=1
        num_samples = 16000  # 1 second at 16kHz

        # First SincNet layer with stride=10
        frames_after_sinc = receptive_field.conv1d_num_frames(
            num_samples, kernel_size=251, stride=10, padding=0, dilation=1
        )

        # MaxPool with kernel=3, stride=3
        frames_after_pool1 = frames_after_sinc // 3

        # Conv1d with kernel=5, stride=1
        frames_after_conv1 = receptive_field.conv1d_num_frames(
            frames_after_pool1, kernel_size=5, stride=1, padding=0, dilation=1
        )

        # Second MaxPool with kernel=3, stride=3
        frames_after_pool2 = frames_after_conv1 // 3

        # Final Conv1d with kernel=5, stride=1
        final_frames = receptive_field.conv1d_num_frames(
            frames_after_pool2, kernel_size=5, stride=1, padding=0, dilation=1
        )

        # Verify we get reasonable output size
        assert final_frames > 0
        assert final_frames < num_samples  # Should be much smaller than input

    def test_receptive_field_consistency(self):
        """Test that receptive field calculations are consistent."""
        # Test that multi-layer functions give same result as chaining
        kernel_sizes = [5, 3, 7]
        strides = [1, 2, 1]
        paddings = [0, 1, 0]
        dilations = [1, 1, 2]

        # Multi-layer calculation
        multi_result = receptive_field.multi_conv_num_frames(
            num_samples=1000,
            kernel_size=kernel_sizes,
            stride=strides,
            padding=paddings,
            dilation=dilations,
        )

        # Chain single-layer calculations
        single_result = 1000
        for k, s, p, d in zip(kernel_sizes, strides, paddings, dilations, strict=True):
            single_result = receptive_field.conv1d_num_frames(
                single_result, kernel_size=k, stride=s, padding=p, dilation=d
            )

        assert multi_result == single_result


class TestMultiConvStartStep:
    """Tests for multi_conv_start_step function."""

    def test_single_layer(self):
        """Single layer: start and step are computed correctly."""
        start, step = receptive_field.multi_conv_start_step(
            kernel_size=[5], stride=[1], padding=[0], dilation=[1]
        )
        assert (start, step) == (2, 1)

    def test_multiple_layers(self):
        """Multiple layers: check iterative computation."""
        start, step = receptive_field.multi_conv_start_step(
            kernel_size=[3, 5], stride=[2, 1], padding=[0, 1], dilation=[1, 1]
        )
        assert (start, step) == (3, 2)

    def test_mismatched_lengths(self):
        """Mismatched parameter lengths should raise ValueError."""
        msg = "kernel_size, stride, padding, dilation must have equal length"
        with pytest.raises(ValueError, match=msg):
            receptive_field.multi_conv_start_step(
                kernel_size=[3, 5], stride=[2], padding=[0, 1], dilation=[1, 1]
            )


class TestFrameCentersSamples:
    """Tests for frame_centers_samples function."""

    def test_list_return_and_length(self):
        """Check list return, length and first/last centers for a simple case."""
        num_samples = 100
        ks = [5]
        st = [1]
        pad = [0]
        dil = [1]

        centers = receptive_field.frame_centers_samples(
            num_samples,
            kernel_size=ks,
            stride=st,
            padding=pad,
            dilation=dil,
            as_numpy=False,
        )

        assert isinstance(centers, list)

        L = receptive_field.multi_conv_num_frames(num_samples, ks, st, pad, dil)
        assert len(centers) == L

        # start should be 2 for a single layer with kernel=5, padding=0
        assert centers[0] == 2
        assert centers[-1] == 2 + (L - 1)

    def test_numpy_return_and_dtype(self):
        """Check numpy return type and integer dtype."""
        centers_np = receptive_field.frame_centers_samples(
            100,
            kernel_size=[5],
            stride=[1],
            padding=[0],
            dilation=[1],
            as_numpy=True,
        )
        assert isinstance(centers_np, np.ndarray)
        assert centers_np.dtype.kind in ("i", "u")
        assert centers_np[0] == 2

    def test_centers_match_start_step(self):
        """Centers vector should match start + step * arange(L)."""
        num_samples = 1000
        ks = [3, 5]
        st = [2, 1]
        pad = [0, 1]
        dil = [1, 1]

        L = receptive_field.multi_conv_num_frames(num_samples, ks, st, pad, dil)
        start, step = receptive_field.multi_conv_start_step(ks, st, pad, dil)

        centers_np = receptive_field.frame_centers_samples(
            num_samples,
            kernel_size=ks,
            stride=st,
            padding=pad,
            dilation=dil,
            as_numpy=True,
        )
        expected = start + step * np.arange(L)
        assert np.array_equal(centers_np, expected.astype(int))
