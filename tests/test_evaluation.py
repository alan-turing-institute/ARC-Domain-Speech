import numpy as np
import pytest

from dr_sad.evaluating import (
    EvaluationMetrics,
    SpeechDetectionEvaluator,
    apply_collar,
    detection_cost_function,
    detection_error_rate,
    frame_accuracy,
)


class TestApplyCollar:
    """Tests for apply_collar function."""

    def test_no_collar(self):
        """Test that apply_collar returns zeros when collar_frames is 0."""
        mask = np.array([0, 0, 1, 1, 1, 0, 0])
        output_evaluation_mask = apply_collar(mask, collar_frames=0)
        expected = np.zeros_like(mask)
        np.testing.assert_array_equal(output_evaluation_mask, expected)

    def test_identifies_boundaries(self):
        """Test that apply_collar identifies and dilates boundaries."""
        mask = np.array([0, 0, 1, 1, 1, 0, 0])
        output_evaluation_mask = apply_collar(mask, collar_frames=1)
        # Boundaries at indices 2 (0->1) and 5 (1->0)
        # With collar_frames=1, should dilate by 1 on each side
        # Expected: collar at indices 1,2,3 and 4,5,6
        expected = np.array([0, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(output_evaluation_mask, expected)

    def test_negative_collar(self):
        """Test that negative collar_frames returns zeros."""
        mask = np.array([0, 0, 1, 1, 1, 0, 0])
        expected_error_msg = "collar_frames must be non-negative, did you mean 5?"
        with pytest.raises(ValueError, match=expected_error_msg):
            _ = apply_collar(mask, collar_frames=-5)


class TestDetectionErrorRate:
    """Tests for detection_error_rate function."""

    def test_perfect_prediction(self):
        """Test DER with perfect predictions."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 0, 1, 1, 1, 0, 0])
        result = detection_error_rate(ground_truth, predictions)

        assert result["der"] == 0.0
        assert result["false_alarm_rate"] == 0.0
        assert result["missed_speech_rate"] == 0.0

    def test_with_errors(self):
        """Test DER with false alarms and missed speech."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 1, 1, 1, 0, 0, 0])  # 1 FA, 1 miss
        result = detection_error_rate(ground_truth, predictions)

        # Total speech frames = 3, non-speech frames = 4
        # False alarm = 1, Missed = 1
        # DER = (1 + 1) / 3 = 0.667
        # FA rate = 1 / 4 = 0.25 (1 FA out of 4 non-speech frames)
        # Miss rate = 1 / 3 = 0.333 (1 miss out of 3 speech frames)
        assert result["der"] == pytest.approx(2 / 3, rel=1e-3)
        assert result["false_alarm_rate"] == pytest.approx(1 / 4, rel=1e-3)
        assert result["missed_speech_rate"] == pytest.approx(1 / 3, rel=1e-3)

    def test_with_collar(self):
        """Test DER with collar ignoring boundary errors."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 1, 1, 1, 1, 0, 0])  # Error at boundary

        # Without collar - should have errors
        result_no_collar = detection_error_rate(
            ground_truth, predictions, collar_frames=0
        )
        assert result_no_collar["der"] > 0

        # With collar - boundary errors ignored
        result_with_collar = detection_error_rate(
            ground_truth, predictions, collar_frames=1
        )
        # DER should be lower or same with collar
        assert result_with_collar["der"] <= result_no_collar["der"]


class TestFrameAccuracy:
    """Tests for frame_accuracy function."""

    def test_perfect(self):
        """Test frame accuracy with perfect predictions."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.3])
        accuracy = frame_accuracy(predictions, ground_truth, threshold=0.5)
        assert accuracy == 1.0

    def test_with_errors(self):
        """Test frame accuracy with some errors."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0.1, 0.7, 0.9, 0.8, 0.2, 0.1, 0.3])  # 2 errors
        accuracy = frame_accuracy(predictions, ground_truth, threshold=0.5)
        assert accuracy == pytest.approx(5 / 7, rel=1e-3)

    def test_all_wrong(self):
        """Test frame accuracy with all predictions wrong."""
        ground_truth = np.array([0, 0, 0, 0, 0])
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        accuracy = frame_accuracy(predictions, ground_truth, threshold=0.5)
        assert accuracy == 0.0


class TestDetectionCostFunction:
    """Tests for detection_cost_function."""

    def test_perfect_prediction(self):
        """Test DCF with perfect predictions."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 0, 1, 1, 1, 0, 0])
        dcf = detection_cost_function(ground_truth, predictions)
        assert dcf == 0.0

    def test_with_errors(self):
        """Test DCF calculation with errors."""
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 1, 1, 1, 0, 0, 0])  # 1 FA, 1 miss
        dcf = detection_cost_function(ground_truth, predictions)

        # FA rate = 1/4 = 0.25, Miss rate = 1/3 = 0.333
        # DCF = 0.75 * miss_rate + 0.25 * fa_rate
        # DCF = 0.75 * (1/3) + 0.25 * (1/4) = 0.25 + 0.0625 = 0.3125
        expected_dcf = 0.75 * (1 / 3) + 0.25 * (1 / 4)
        assert dcf == pytest.approx(expected_dcf, rel=1e-3)

    def test_all_false_alarms(self):
        """Test DCF with only false alarms."""
        ground_truth = np.array([0, 0, 0, 0, 0])
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        dcf = detection_cost_function(ground_truth, predictions)
        # FA rate = 1.0, Miss rate = undefined (no speech)
        # Should handle gracefully
        assert dcf >= 0.0


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_creation(self):
        """Test creating EvaluationMetrics object."""
        metrics = EvaluationMetrics(
            der=0.15,
            false_alarm_rate=0.08,
            missed_speech_rate=0.07,
            accuracy=0.92,
            detection_cost_function=0.10,
        )
        assert metrics.der == 0.15
        assert metrics.accuracy == 0.92
        assert metrics.detection_cost_function == 0.10

    def test_to_dict(self):
        """Test converting EvaluationMetrics to dict."""
        metrics = EvaluationMetrics(
            der=0.15,
            false_alarm_rate=0.08,
            missed_speech_rate=0.07,
            accuracy=0.92,
            detection_cost_function=0.10,
        )
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["der"] == 0.15
        assert metrics_dict["accuracy"] == 0.92
        assert metrics_dict["detection_cost_function"] == 0.10

    def test_format_for_plot(self):
        """Test formatting metrics for plot display."""
        metrics = EvaluationMetrics(
            der=0.15,
            false_alarm_rate=0.08,
            missed_speech_rate=0.07,
            accuracy=0.92,
            detection_cost_function=0.10,
        )
        plot_text = metrics.format_for_plot()
        assert isinstance(plot_text, str)
        assert "DER" in plot_text
        assert "DCF" in plot_text
        assert "Acc" in plot_text


class TestSpeechDetectionEvaluator:
    """Tests for SpeechDetectionEvaluator class."""

    def test_calculate_detection_error_rate(self):
        """Test evaluator's DER calculation method."""
        evaluator = SpeechDetectionEvaluator(collar_frames=0, detection_threshold=0.5)
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 0, 1, 1, 1, 0, 0])

        result = evaluator.calculate_detection_error_rate(ground_truth, predictions)
        assert result["der"] == 0.0

    def test_calculate_frame_accuracy(self):
        """Test evaluator's frame accuracy calculation method."""
        evaluator = SpeechDetectionEvaluator(collar_frames=0, detection_threshold=0.5)
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.3])

        accuracy = evaluator.calculate_frame_accuracy(ground_truth, predictions)
        assert accuracy == 1.0

    def test_calculate_detection_cost_function(self):
        """Test evaluator's DCF calculation method."""
        evaluator = SpeechDetectionEvaluator(collar_frames=0, detection_threshold=0.5)
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0, 0, 1, 1, 1, 0, 0])

        dcf = evaluator.calculate_detection_cost_function(ground_truth, predictions)
        assert dcf == 0.0

    def test_evaluate(self):
        """Test evaluator's full evaluation method."""
        evaluator = SpeechDetectionEvaluator(collar_frames=0, detection_threshold=0.5)
        ground_truth = np.array([0, 0, 1, 1, 1, 0, 0])
        predictions = np.array([0.1, 0.7, 0.9, 0.8, 0.2, 0.1, 0.3])  # 2 errors

        metrics = evaluator.evaluate(ground_truth, predictions)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.der == pytest.approx(
            2 / 3, rel=1e-3
        )  # (1 FA + 1 Miss) / 3 speech frames = 2/3
        assert metrics.accuracy == pytest.approx(
            5 / 7, rel=1e-3
        )  # 5 correct out of 7 frames
        assert metrics.false_alarm_rate == pytest.approx(
            1 / 4, rel=1e-3
        )  # 1 false alarm out of 4 non-speech frames
        assert metrics.missed_speech_rate == pytest.approx(
            1 / 3, rel=1e-3
        )  # 1 missed out of 3 speech frames
        assert metrics.detection_cost_function == pytest.approx(
            0.75 * (1 / 3) + 0.25 * (1 / 4), rel=1e-3
        )  # DCF = 0.75 * miss_rate + 0.25 * fa_rate
