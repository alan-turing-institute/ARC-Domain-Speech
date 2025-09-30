import numpy as np
import pytest

from dr_sad import annotation as an


class TestSpeakingMap:
    """Tests for the speaking_map function."""

    def test_basic_functionality_with_list(self):
        """Test basic functionality with list input."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        annotations = [(1.0, 3.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([0, 1, 1, 1, 0, 0], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.int16

    def test_basic_functionality_with_numpy_array(self):
        """Test basic functionality with numpy array input."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        annotations = [(1.0, 3.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([0, 1, 1, 1, 0, 0], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    def test_empty_annotations(self):
        """Test with no annotations - should return all zeros."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        annotations: list[tuple[float, float]] = []

        result = an.speaking_map(timestamps, annotations)
        expected = np.zeros(5, dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    def test_multiple_annotations(self):
        """Test with multiple non-overlapping annotations."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        annotations = [(1.0, 2.0), (4.0, 6.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    def test_overlapping_annotations(self):
        """Test with overlapping annotations."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        annotations = [(1.0, 3.0), (2.0, 4.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([0, 1, 1, 1, 1, 0], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    def test_annotation_at_boundaries(self):
        """Test annotations that exactly match timestamp boundaries."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        annotations = [(0.0, 2.0), (3.0, 4.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([1, 1, 1, 1, 1], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    def test_single_timestamp(self):
        """Test with single timestamp."""
        timestamps = [2.0]
        annotations = [(1.0, 3.0)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([1], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)

    # Error condition tests
    def test_invalid_timestamps_conversion(self):
        """Test error when timestamps cannot be converted to numpy array."""
        timestamps = ["invalid", "data"]
        annotations = [(1.0, 2.0)]

        with pytest.raises(
            ValueError, match="Could not convert timestamps to numpy array"
        ):
            an.speaking_map(timestamps, annotations)

    def test_multidimensional_timestamps(self):
        """Test error with multidimensional timestamp array."""
        timestamps = np.array([[1.0, 2.0], [3.0, 4.0]])
        annotations = [(1.0, 2.0)]

        with pytest.raises(ValueError, match="Timestamps must be a 1D array"):
            an.speaking_map(timestamps, annotations)

    def test_empty_timestamps(self):
        """Test error with empty timestamps array."""
        timestamps: list[float] = []
        annotations = [(1.0, 2.0)]

        with pytest.raises(ValueError, match="Timestamps array is empty"):
            an.speaking_map(timestamps, annotations)

    def test_empty_numpy_timestamps(self):
        """Test error with empty numpy timestamps array."""
        timestamps = np.array([])
        annotations = [(1.0, 2.0)]

        with pytest.raises(ValueError, match="Timestamps array is empty"):
            an.speaking_map(timestamps, annotations)

    def test_invalid_annotation_length(self):
        """Test error with annotation that doesn't have exactly 2 elements."""
        timestamps = [1.0, 2.0, 3.0]
        annotations = [(1.0, 2.0, 3.0)]  # Too many elements

        with pytest.raises(ValueError, match="Each annotation must be a tuple of"):
            an.speaking_map(timestamps, annotations)  # type: ignore[arg-type]

    def test_invalid_annotation_single_element(self):
        """Test error with annotation that has only one element."""
        timestamps = [1.0, 2.0, 3.0]
        annotations = [(1.0,)]  # Only one element

        with pytest.raises(ValueError, match="Each annotation must be a tuple of"):
            an.speaking_map(timestamps, annotations)  # type: ignore[arg-type]

    def test_invalid_annotation_order(self):
        """Test error when annotation start >= end."""
        timestamps = [1.0, 2.0, 3.0]
        annotations = [(2.0, 1.0)]  # start > end

        with pytest.raises(
            ValueError, match="Annotation start time must be less than end time"
        ):
            an.speaking_map(timestamps, annotations)

    def test_invalid_annotation_equal_times(self):
        """Test error when annotation start == end."""
        timestamps = [1.0, 2.0, 3.0]
        annotations = [(2.0, 2.0)]  # start == end

        with pytest.raises(
            ValueError, match="Annotation start time must be less than end time"
        ):
            an.speaking_map(timestamps, annotations)

    def test_unsorted_timestamps(self):
        """Test behavior with unsorted timestamps."""
        timestamps = [3.0, 1.0, 4.0, 2.0]
        annotations = [(1.5, 3.5)]

        result = an.speaking_map(timestamps, annotations)
        expected = np.array([1, 0, 0, 1], dtype=np.int16)

        np.testing.assert_array_equal(result, expected)
