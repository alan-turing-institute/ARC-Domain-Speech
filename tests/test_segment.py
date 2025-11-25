import numpy as np
import pytest

from dr_sad.segment import binarise, segment_scores, segment_times


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


class TestSegmentTimes:
    def test_simple(self):
        input_array = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        expected_times = [(1.25, 3.25)]
        output_times = segment_times(input_array, 1, 0.5)
        np.testing.assert_almost_equal(output_times, expected_times)

    def test_no_segments(self):
        input_array = np.array([0, 0, 0, 0], dtype=bool)
        expected_times: list[tuple[float, float]] = []
        output_times = segment_times(input_array, 1, 0.5)
        np.testing.assert_almost_equal(output_times, expected_times)

    def test_starting_on(self):
        input_array = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        expected_times = [(0.0, 1.75)]
        output_times = segment_times(input_array, 1, 0.5)
        np.testing.assert_almost_equal(output_times, expected_times)

    def test_ending_on(self):
        input_array = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        expected_times = [(2.75, 4.5)]
        output_times = segment_times(input_array, 1, 0.5)
        np.testing.assert_almost_equal(output_times, expected_times)

    def test_multiple_segments(self):
        input_array = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
        expected_times = [(1.25, 2.25), (2.75, 4.25)]
        output_times = segment_times(input_array, 1, 0.5)
        np.testing.assert_almost_equal(output_times, expected_times)

    def test_different_time_step(self):
        input_array = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
        expected_times = [(1.125, 1.875)]
        output_times = segment_times(input_array, 1.0, 0.25)
        np.testing.assert_almost_equal(output_times, expected_times)


class TestSegmentScores:
    def test_simple(self):
        predicted_segments = [(1.0, 2.0), (3.0, 4.0)]
        reference_segments = [(1.0, 2.0), (10.0, 12.0)]
        tolerance = 0.1
        expec_tp, expec_fp, expec_fn = 1, 1, 1
        tp, fp, fn = segment_scores(predicted_segments, reference_segments, tolerance)
        assert (tp, fp, fn) == (expec_tp, expec_fp, expec_fn)

    def test_tolerance(self):
        predicted_segments = [(1.0, 2.0), (3.2, 4.0)]
        reference_segments = [(1.05, 1.95), (3.0, 4.0)]
        tolerance = 0.1
        expec_tp, expec_fp, expec_fn = 1, 1, 1
        tp, fp, fn = segment_scores(predicted_segments, reference_segments, tolerance)
        assert (tp, fp, fn) == (expec_tp, expec_fp, expec_fn)

    def test_no_matches(self):
        predicted_segments = [(5.0, 6.0)]
        reference_segments = [(1.0, 2.0)]
        tolerance = 0.1
        expec_tp, expec_fp, expec_fn = 0, 1, 1
        tp, fp, fn = segment_scores(predicted_segments, reference_segments, tolerance)
        assert (tp, fp, fn) == (expec_tp, expec_fp, expec_fn)

    def test_no_predictions(self):
        predicted_segments: list[tuple[float, float]] = []
        reference_segments = [(1.0, 2.0)]
        tolerance = 0.1
        expec_tp, expec_fp, expec_fn = 0, 0, 1
        tp, fp, fn = segment_scores(predicted_segments, reference_segments, tolerance)
        assert (tp, fp, fn) == (expec_tp, expec_fp, expec_fn)

    def test_no_references(self):
        predicted_segments = [(1.0, 2.0)]
        reference_segments: list[tuple[float, float]] = []
        tolerance = 0.1
        expec_tp, expec_fp, expec_fn = 0, 1, 0
        tp, fp, fn = segment_scores(predicted_segments, reference_segments, tolerance)
        assert (tp, fp, fn) == (expec_tp, expec_fp, expec_fn)
