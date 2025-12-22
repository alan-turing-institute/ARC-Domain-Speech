import numpy as np
import pytest
from scipy import optimize

from dr_sad.segment import (
    DiffEvolOptimizer,
    SegmentEvaluator,
    binarise,
    dataset_to_segments,
    f1_score_set,
    segment_scores,
    segment_times,
)


class TestBinarise:
    def test_simple(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 0, 1, 0], dtype=bool)
        output = binarise(input_array)
        assert np.array_equal(output, expected_output)

    def test_on_threshold(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 1, 1, 0], dtype=bool)
        output = binarise(input_array, speech_threshold=0.35)
        assert np.array_equal(output, expected_output)

    def test_with_off_threshold(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])
        expected_output = np.array([0, 1, 1, 1, 0], dtype=bool)
        output = binarise(input_array, speech_threshold=0.4, gap_threshold=0.2)
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
            binarise(input_array, speech_threshold=0.5)

    def test_invalid_thresholds(self):
        input_array = np.array([0.1, 0.6, 0.4, 0.8, 0.2])

        with pytest.raises(ValueError, match="gap_threshold"):
            binarise(input_array, speech_threshold=0.5, gap_threshold=-0.2)

    def test_empty_array(self):
        input_array = np.array([], dtype=float)
        with pytest.raises(ValueError, match="Input array is empty"):
            binarise(input_array)


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


class TestF1ScoreSet:
    def test_simple(self):
        predicted_segments = [[(1.0, 2.0)], [(3.0, 4.0)]]
        reference_segments = [[(1.0, 2.0)], [(10.0, 12.0)]]
        tolerance = 0.1
        expected_f1 = 0.5
        f1 = f1_score_set(predicted_segments, reference_segments, tolerance)
        np.testing.assert_almost_equal(f1, expected_f1)

    def test_no_predictions(self):
        predicted_segments: list[list[tuple[float, float]]] = [[], []]
        reference_segments = [[(1.0, 2.0)], [(3.0, 4.0)]]
        tolerance = 0.1
        expected_f1 = 0.0
        f1 = f1_score_set(predicted_segments, reference_segments, tolerance)
        np.testing.assert_almost_equal(f1, expected_f1)

    def test_no_references(self):
        predicted_segments = [[(1.0, 2.0)], [(3.0, 4.0)]]
        reference_segments: list[list[tuple[float, float]]] = [[], []]
        tolerance = 0.1
        expected_f1 = 0.0
        f1 = f1_score_set(predicted_segments, reference_segments, tolerance)
        np.testing.assert_almost_equal(f1, expected_f1)

    def test_perfect_match(self):
        predicted_segments = [[(1.0, 2.0)], [(3.0, 4.0)]]
        reference_segments = [[(1.01, 2.02)], [(3.05, 3.95)]]
        tolerance = 0.1
        expected_f1 = 1.0
        f1 = f1_score_set(predicted_segments, reference_segments, tolerance)
        np.testing.assert_almost_equal(f1, expected_f1)

    def test_all_wrong(self):
        predicted_segments = [[(5.0, 6.0)], [(7.0, 8.0)]]
        reference_segments = [[(1.0, 2.0)], [(3.0, 4.0)]]
        tolerance = 0.1
        expected_f1 = 0.0
        f1 = f1_score_set(predicted_segments, reference_segments, tolerance)
        np.testing.assert_almost_equal(f1, expected_f1)


class TestDatasetToSegments:
    def test_simple(self):
        predictions = [np.array([0.1, 0.6, 0.7, 0.8, 0.2])]
        time_start = 0.0
        time_step = 1.0
        expected_segments = [[(0.5, 3.5)]]
        segments = dataset_to_segments(predictions, time_start, time_step)
        for seg_list, exp_list in zip(segments, expected_segments, strict=True):
            np.testing.assert_almost_equal(seg_list, exp_list)

    def test_multiple_samples(self):
        predictions = [
            np.array([0.1, 0.6, 0.7, 0.8, 0.2]),
            np.array([0.7, 0.2, 0.9, 0.8]),
        ]
        time_start = 0.0
        time_step = 1.0
        expected_segments = [[(0.5, 3.5)], [(0.0, 0.5), (1.5, 3.0)]]
        segments = dataset_to_segments(predictions, time_start, time_step)
        for seg_list, exp_list in zip(segments, expected_segments, strict=True):
            np.testing.assert_almost_equal(seg_list, exp_list)


class TestSegmentEvaluator:
    def test_init_simple(self):
        prediction_set = {"test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2])}
        reference_set = {"test01": [(1.5, 4.5)]}

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
        )
        assert len(seg_eval.keys) == 1
        assert len(seg_eval.predictions) == 1
        assert len(seg_eval.references) == 1
        assert seg_eval.tolerance == 0.5
        assert seg_eval.time_start == 1.0
        assert seg_eval.time_step == 1.0

    def test_init_raises_on_key_mismatch(self):
        prediction_set = {"test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2])}
        reference_set = {"test02": [(1.5, 4.5)]}

        with pytest.raises(ValueError, match="Key test01 not found in reference set"):
            SegmentEvaluator(prediction_set, reference_set, 1.0, 1.0, 0.2)

    def test_get_parameters(self):
        prediction_set = {"test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2])}
        reference_set = {"test01": [(1.5, 4.5)]}

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.3,
        )

        params = seg_eval.get_parameters()
        expected_params = {
            "speech_threshold": 0.5,
            "gap_threshold": None,
            "min_duration_off": None,
            "min_duration_on": None,
        }
        assert params == expected_params

    def test_set_parameters(self):
        prediction_set = {"test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2])}
        reference_set = {"test01": [(1.5, 4.5)]}

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.3,
        )

        seg_eval.set_parameters(
            speech_threshold=0.6,
            gap_threshold=0.4,
            min_duration_off=0.5,
            min_duration_on=1.0,
        )

        params = seg_eval.get_parameters()
        expected_params = {
            "speech_threshold": 0.6,
            "gap_threshold": 0.4,
            "min_duration_off": 0.5,
            "min_duration_on": 1.0,
        }
        assert params == expected_params

    def test_generate_segments(self):
        prediction_set = {
            "test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.1]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 3.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.2,
        )

        segments_dict = seg_eval.generate_segments(as_list=False)
        expected_segments_dict = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 4.5)],
        }
        assert isinstance(segments_dict, dict)
        assert segments_dict.keys() == expected_segments_dict.keys()
        for key in seg_eval.keys:
            np.testing.assert_almost_equal(
                segments_dict[key],
                expected_segments_dict[key],
            )

        segments_list = seg_eval.generate_segments(as_list=True)
        expected_segments_list = [
            [(1.5, 4.5)],
            [(0.0, 1.5), (2.5, 4.5)],
        ]
        assert isinstance(segments_list, list)
        assert len(segments_list) == len(expected_segments_list)
        for i in range(len(expected_segments_list)):
            np.testing.assert_almost_equal(
                segments_list[i],
                expected_segments_list[i],
            )

    def test_f1_score(self):
        prediction_set = {
            "test01": np.array([0.1, 0.6, 0.7, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.1]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 3.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
        )

        f1_predicted = seg_eval.f1_score()
        expected_f1 = 2 / 3
        assert np.isclose(f1_predicted, expected_f1)

    def test_parameters_effect_f1(self):
        prediction_set = {
            "test01": np.array([0.1, 0.8, 0.9, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.47]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 6.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
        )

        f1_default = seg_eval.f1_score()
        expected_f1_default = 2 / 3
        assert np.isclose(f1_default, expected_f1_default)

        seg_eval.set_parameters(speech_threshold=0.5, gap_threshold=0.1)

        f1_updated = seg_eval.f1_score()
        expected_f1_updated = 1.0
        assert np.isclose(f1_updated, expected_f1_updated)

    def test_optimize_parameters(self):
        prediction_set = {
            "test01": np.array([0.1, 0.8, 0.9, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.47]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 6.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
        )
        original_params = seg_eval.get_parameters()

        result = seg_eval.optimise_parameters(
            speech_threshold=True,
            gap_threshold=True,
            min_duration_off=True,
            min_duration_on=True,
        )

        new_params = seg_eval.get_parameters()

        assert isinstance(result, optimize.OptimizeResult)
        assert result.success
        assert new_params != original_params

    def test_optimize_one_parameter(self):
        prediction_set = {
            "test01": np.array([0.1, 0.8, 0.9, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.47]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 6.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
            speech_threshold=0.55,
            gap_threshold=0.5,  # number too big
        )
        original_params = seg_eval.get_parameters()

        result = seg_eval.optimise_parameters(
            speech_threshold=False,
            gap_threshold=True,
            min_duration_off=False,
            min_duration_on=False,
            maxiter=3,
        )

        new_params = seg_eval.get_parameters()

        assert isinstance(result, optimize.OptimizeResult)
        assert new_params != original_params
        assert new_params["speech_threshold"] == original_params["speech_threshold"]
        assert new_params["gap_threshold"] != original_params["gap_threshold"]
        assert new_params["min_duration_off"] == original_params["min_duration_off"]

    def test_optimise_diff_evol(self):
        prediction_set = {
            "test01": np.array([0.1, 0.8, 0.9, 0.8, 0.2]),
            "test02": np.array([0.7, 0.2, 0.9, 0.8, 0.47]),
        }
        reference_set = {
            "test01": [(1.5, 4.5)],
            "test02": [(0.0, 1.5), (2.5, 6.0)],
        }

        seg_eval = SegmentEvaluator(
            prediction_set,
            reference_set,
            time_start=1.0,
            time_step=1.0,
            tolerance=0.5,
        )
        original_params = seg_eval.get_parameters()

        result = seg_eval.optimise_diff_evol(
            speech_threshold=True,
            gap_threshold=True,
            min_duration_off=True,
            min_duration_on=True,
            num_workers=2,
            maxiter=2,
        )

        new_params = seg_eval.get_parameters()

        assert isinstance(result, optimize.OptimizeResult)
        assert new_params != original_params


class TestDiffEvolOptimizer:
    def test_init_valid_parameters(self):
        """Test successful initialization with valid parameters."""
        predictions = [np.array([0.1, 0.8, 0.9, 0.2])]
        references = [[(0.5, 1.5)]]
        start_parameters: dict[str, float | None] = {
            "speech_threshold": 0.5,
            "gap_threshold": 0.1,
            "min_duration_off": 1.0,
            "min_duration_on": 0.5,
        }
        optimise_parameters = ["speech_threshold", "gap_threshold"]

        optimizer = DiffEvolOptimizer(
            predictions=predictions,
            references=references,
            start_parameters=start_parameters,
            optimise_parameters=optimise_parameters,
            time_start=0.0,
            time_step=0.5,
            tolerance=0.1,
        )

        assert optimizer.predictions == predictions
        assert optimizer.references == references
        assert optimizer.parameters == start_parameters
        assert optimizer.optimise_parameters == optimise_parameters
        assert optimizer.time_start == 0.0
        assert optimizer.time_step == 0.5
        assert optimizer.tolerance == 0.1

    def test_call_single_parameter(self):
        """Test the __call__ method with a single optimization parameter."""
        predictions = [np.array([0.1, 0.8, 0.9, 0.2, 0.3])]
        references = [[(0.5, 1.5)]]
        start_parameters: dict[str, float | None] = {
            "speech_threshold": 0.5,
            "gap_threshold": 0.1,
            "min_duration_off": 1.0,
            "min_duration_on": 0.5,
        }
        optimise_parameters = ["speech_threshold"]

        optimizer = DiffEvolOptimizer(
            predictions=predictions,
            references=references,
            start_parameters=start_parameters,
            optimise_parameters=optimise_parameters,
            time_start=0.0,
            time_step=0.5,
            tolerance=0.1,
        )

        # Call with new speech_threshold value
        result = optimizer([0.7])

        # Check that parameter was updated
        assert optimizer.parameters["speech_threshold"] == 0.7
        # Other parameters should remain unchanged
        assert optimizer.parameters["gap_threshold"] == 0.1
        assert optimizer.parameters["min_duration_off"] == 1.0
        assert optimizer.parameters["min_duration_on"] == 0.5

        # Result should be a float (negative F1 score)
        assert isinstance(result, float)
        assert result <= 0.0  # F1 score is negated for minimization

    def test_call_multiple_parameters(self):
        """Test the __call__ method with multiple optimization parameters."""
        predictions = [np.array([0.1, 0.8, 0.9, 0.2, 0.3])]
        references = [[(0.5, 1.5)]]
        start_parameters: dict[str, float | None] = {
            "speech_threshold": 0.5,
            "gap_threshold": 0.1,
            "min_duration_off": 1.0,
            "min_duration_on": 0.5,
        }
        optimise_parameters = ["speech_threshold", "gap_threshold", "min_duration_off"]

        optimizer = DiffEvolOptimizer(
            predictions=predictions,
            references=references,
            start_parameters=start_parameters,
            optimise_parameters=optimise_parameters,
            time_start=0.0,
            time_step=0.5,
            tolerance=0.1,
        )

        # Call with new parameter values
        new_values = [0.6, 0.2, 0.8]
        result = optimizer(new_values)

        # Check that all parameters were updated
        assert optimizer.parameters["speech_threshold"] == 0.6
        assert optimizer.parameters["gap_threshold"] == 0.2
        assert optimizer.parameters["min_duration_off"] == 0.8
        # Unchanged parameter should remain the same
        assert optimizer.parameters["min_duration_on"] == 0.5

        # Result should be a float (negative F1 score)
        assert isinstance(result, float)
        assert result <= 0.0
