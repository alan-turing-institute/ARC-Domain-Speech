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

    def test_min_duration_off(self):
        input_array = np.array([0.8, 0.6, 0.4, 0.7, 0.8, 0.2, 0.1, 0.1])
        expected_output = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        output = binarise(input_array, min_duration_off=2)
        assert np.array_equal(output, expected_output)

    def test_min_duration_on(self):
        input_array = np.array([0.1, 0.6, 0.2, 0.4, 0.2, 0.9, 0.8, 0.7])
        expected_output = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        output = binarise(input_array, min_duration_on=2)
        assert np.array_equal(output, expected_output)

    def test_min_duration_off_start_end(self):
        input_array = np.array([0.2, 0.6, 0.7, 0.2, 0.1, 0.8, 0.9, 0.8, 0.2])
        expected_output = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        output = binarise(input_array, min_duration_off=2)
        assert np.array_equal(output, expected_output)

    def test_min_duration_on_start_end(self):
        input_array = np.array([0.6, 0.7, 0.2, 0.1, 0.4, 0.9, 0.9, 0.8])
        expected_output = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        output = binarise(input_array, min_duration_on=3)
        assert np.array_equal(output, expected_output)

    def test_invalid_input(self):
        input_array = np.array([[0.1, 0.6], [0.4, 0.8]])

        with pytest.raises(ValueError, match="Input array must be 1D"):
            binarise(input_array, on_threshold=0.5)

    def test_invalid_thresholds(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])

        with pytest.raises(ValueError, match="On threshold must"):
            binarise(input_array, on_threshold=0.3, off_threshold=0.5)
