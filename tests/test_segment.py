import numpy as np
import pytest

from dr_sad.segment import binarise


class TestBinarise:
    def test_simple(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 0, 1, 0], dtype=bool)
        output = binarise(input_array)
        assert np.array_equal(output, expected_output)

    def test_on_threshold(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 1, 1, 0], dtype=bool)
        output = binarise(input_array, on_threshold=0.35)
        assert np.array_equal(output, expected_output)

    def test_with_off_threshold(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 1, 1, 0], dtype=bool)
        output = binarise(input_array, on_threshold=0.5, off_threshold=0.3)
        assert np.array_equal(output, expected_output)

    def test_invalid_input(self):
        input_array = np.array([[0.1, 0.6], [0.4, 0.8]])

        with pytest.raises(ValueError, match="Input array must be 1D"):
            binarise(input_array, on_threshold=0.5)

    def test_invalid_thresholds(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])

        with pytest.raises(ValueError, match="On threshold must"):
            binarise(input_array, on_threshold=0.3, off_threshold=0.5)
