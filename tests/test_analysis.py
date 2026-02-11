import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from dr_sad import analysis
from dr_sad.analysis import (
    downsample_to_prediction_frames,
    inverse_weightings_by_domain,
    load_annotations,
    load_audio_and_annotations,
)


@pytest.fixture()
def sample_data_table():
    """Create a sample data table for testing."""
    data = {
        "file_id": ["file1", "file2", "file3", "file4", "file5", "file6"],
        "domain": [
            "webvideo",
            "webvideo",
            "webvideo",
            "clinical",
            "clinical",
            "restaurant",
        ],
        "lang": ["english"] * 6,
        "source": ["librivox"] * 6,
    }
    return pd.DataFrame(data)


@pytest.fixture()
def temp_data_table_file(sample_data_table, tmp_path):
    """Create a temporary TSV file with sample data using pytest tmp_path."""
    temp_file = tmp_path / "test_data.tbl"
    sample_data_table.to_csv(temp_file, sep="\t", index=False)
    return temp_file


class TestLoadAnnotations:
    def load_valid_file(self, test_dataset):
        """Test loading a valid RTTM file."""
        speech_segments = load_annotations(
            file_id="TEST_0001", data_dir=str(test_dataset)
        )

        assert len(speech_segments) == 2
        assert speech_segments[0] == (0.5, 1.0)  # start at 0.5s, duration 0.5s
        assert speech_segments[1] == (1.5, 2.0)  # start at 1.5s, duration 0.5s


class TestLoadAudioAndAnnotations:
    """Tests for _load_audio_and_annotations function."""

    def test_load_valid_file(self, example_dataset):
        """Test loading a valid audio file with annotations."""
        audio, sample_rate, speech_segments = load_audio_and_annotations(
            file_id="TEST_0001", data_dir=str(example_dataset)
        )

        # Check audio properties
        assert isinstance(audio, np.ndarray)
        assert sample_rate == 16000
        assert len(audio) == int(2.5 * sample_rate)  # 2.5 seconds at 16kHz

        # Check speech segments
        assert len(speech_segments) == 2
        assert speech_segments[0] == (0.5, 1.0)  # start at 0.5s, duration 0.5s
        assert speech_segments[1] == (1.5, 2.0)  # start at 1.5s, duration 0.5s

    def test_load_multiple_files(self, example_dataset):
        """Test loading multiple different files."""
        file_ids = ["TEST_0001", "TEST_0005", "TEST_0010"]

        for file_id in file_ids:
            audio, sample_rate, speech_segments = load_audio_and_annotations(
                file_id=file_id, data_dir=str(example_dataset)
            )
            assert len(audio) > 0
            assert sample_rate == 16000
            assert len(speech_segments) == 2

    def test_load_nonexistent_file(self, example_dataset):
        """Test that loading a nonexistent file raises an error."""
        with pytest.raises((FileNotFoundError, sf.LibsndfileError)):
            load_audio_and_annotations(
                file_id="NONEXISTENT", data_dir=str(example_dataset)
            )

    def test_rttm_parsing_format(self):
        """Test correct parsing of RTTM file format."""
        # Create a custom RTTM file with specific segments
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            flac_dir = dataset_dir / "flac"
            rttm_dir = dataset_dir / "rttm"
            flac_dir.mkdir()
            rttm_dir.mkdir()

            # Create audio file
            sample_rate = 16000
            audio = np.random.uniform(-0.1, 0.1, sample_rate * 3).astype(np.float32)
            audio_path = flac_dir / "TEST.flac"
            sf.write(audio_path, audio, sample_rate)

            # Create RTTM with multiple segments
            rttm_path = rttm_dir / "TEST.rttm"
            with open(rttm_path, "w") as f:
                f.write("SPEAKER TEST 1 0.100 0.200 <NA> <NA> spk1 <NA> <NA>\n")
                f.write("SPEAKER TEST 1 0.500 0.300 <NA> <NA> spk2 <NA> <NA>\n")
                f.write("SPEAKER TEST 1 2.000 0.500 <NA> <NA> spk1 <NA> <NA>\n")

            audio, sample_rate, segments = load_audio_and_annotations(
                file_id="TEST", data_dir=str(dataset_dir)
            )

            assert len(segments) == 3
            # Use pytest.approx for floating point comparison
            assert segments[0] == pytest.approx((0.1, 0.3))  # start + duration
            assert segments[1] == pytest.approx((0.5, 0.8))
            assert segments[2] == pytest.approx((2.0, 2.5))


class TestDownsampleToPredictionFrames:
    """Tests for _downsample_to_prediction_frames function."""

    def test_binary_signal_downsampling(self):
        """Test downsampling a binary signal preserves binary values."""
        # Create a binary signal
        signal = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
        prediction_length = 6

        downsampled = downsample_to_prediction_frames(
            signal, prediction_length, is_binary=True
        )

        assert len(downsampled) == prediction_length
        # All values should be 0 or 1
        assert np.all((downsampled == 0) | (downsampled == 1))

    def test_continuous_signal_downsampling(self):
        """Test downsampling a continuous signal."""
        # Create a sine wave
        signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
        prediction_length = 100

        downsampled = downsample_to_prediction_frames(
            signal, prediction_length, is_binary=False
        )

        assert len(downsampled) == prediction_length
        # Should preserve approximate range
        assert np.min(downsampled) >= -1.1
        assert np.max(downsampled) <= 1.1

    def test_downsampling_preserves_distribution(self):
        """Test that binary downsampling preserves approximate distribution."""
        # Create signal with 50% speech activity
        signal = np.zeros(1000)
        signal[250:750] = 1.0  # 50% of signal is speech

        downsampled = downsample_to_prediction_frames(signal, 100, is_binary=True)

        # Distributions should be approximately preserved (within 10% tolerance)
        original_distribution = np.sum(signal) / len(signal)
        downsampled_distribution = np.sum(downsampled) / len(downsampled)
        assert abs(original_distribution - downsampled_distribution) < 0.01

    def test_extreme_downsampling(self):
        """Test extreme downsampling ratios."""
        # Very large signal downsampled to very small
        signal = np.random.rand(10000)
        prediction_length = 10

        downsampled = downsample_to_prediction_frames(
            signal, prediction_length, is_binary=False
        )

        assert len(downsampled) == prediction_length
        assert not np.any(np.isnan(downsampled))
        assert not np.any(np.isinf(downsampled))

    def test_upsampling_behaviour(self):
        """Test behaviour when target length is larger than input."""
        # Small signal upsampled to larger
        signal = np.array([0, 1, 0, 1, 0], dtype=float)
        prediction_length = 20

        downsampled = downsample_to_prediction_frames(
            signal, prediction_length, is_binary=True
        )

        assert len(downsampled) == prediction_length
        # Binary values should still be preserved
        assert np.all((downsampled == 0) | (downsampled == 1))


class TestInverseWeightingsByDomain:
    """Test cases for inverse_weightings_by_domain function."""

    @pytest.fixture(autouse=True)
    def _mock_domain_settings(self, monkeypatch):
        """Mock DOMAIN_SETTINGS for all tests in this class."""
        mock_settings = {"test": {"domain_column": "domain"}}
        monkeypatch.setattr(analysis, "DOMAIN_SETTINGS", mock_settings)

    def test_basic_functionality(self, temp_data_table_file):
        """Test basic functionality with known data."""
        file_ids = ["file1", "file2", "file3", "file4", "file5", "file6"]

        file_weights = inverse_weightings_by_domain(
            file_ids, temp_data_table_file, data_name="test"
        )

        # Check return structure
        assert isinstance(file_weights, dict)
        for file_id in file_ids:
            assert file_id in file_weights

        # Check file weights mapping
        assert len(file_weights) == len(file_ids)

        # Files from same domain should have same weight
        assert file_weights["file1"] == file_weights["file2"] == file_weights["file3"]
        assert file_weights["file4"] == file_weights["file5"]

    def test_inverse_weighting_calculation(self, temp_data_table_file):
        """Test that inverse weighting calculation is correct."""
        file_ids = ["file1", "file2", "file3", "file4", "file5", "file6"]
        total_files = len(file_ids)

        result = inverse_weightings_by_domain(
            file_ids, temp_data_table_file, data_name="test"
        )

        # Calculate expected weights manually
        # webvideo: 6/3 = 2, clinical: 6/2 = 3, restaurant: 6/1 = 6
        # Total weight: 2 + 3 + 6 = 11
        # Normalized: webvideo: 2/11, clinical: 3/11, restaurant: 6/11

        expected_webvideo = (total_files / 3) / 11
        expected_clinical = (total_files / 2) / 11
        expected_restaurant = (total_files / 1) / 11

        assert abs(result["file1"] - expected_webvideo) < 1e-10
        assert abs(result["file4"] - expected_clinical) < 1e-10
        assert abs(result["file6"] - expected_restaurant) < 1e-10

    def test_subset_of_files(self, temp_data_table_file):
        """Test with only a subset of available files."""
        file_ids = ["file1", "file4", "file6"]  # One from each domain

        result = inverse_weightings_by_domain(
            file_ids, temp_data_table_file, data_name="test"
        )

        # Should have equal counts for each file, so equal weights
        expected_weight = 1 / 3

        for weight in result.values():
            assert abs(weight - expected_weight) < 1e-10

    def test_single_domain(self, temp_data_table_file):
        """Test with files from only one domain."""
        file_ids = ["file1", "file2", "file3"]  # Only webvideo files

        result = inverse_weightings_by_domain(
            file_ids, temp_data_table_file, data_name="test"
        )

        # All files should have the same weight
        expected_file_weight = 1.0
        for weight in result.values():
            assert weight == expected_file_weight

    def test_empty_file_list(self, temp_data_table_file):
        """Test with empty file list."""
        file_ids: list[str] = []

        result = inverse_weightings_by_domain(
            file_ids, temp_data_table_file, data_name="test"
        )

        # All dictionaries should be empty
        assert result == {}

    def test_file_not_found(self):
        """Test with non-existent data table file."""
        file_ids = ["file1", "file2"]
        nonexistent_path = Path("/nonexistent/path/data.tbl")

        with pytest.raises(FileNotFoundError):
            inverse_weightings_by_domain(file_ids, nonexistent_path, data_name="test")
