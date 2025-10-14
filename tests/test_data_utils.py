"""Tests for dr_sad.data.utils module."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dr_sad.data.callhome_utils import DOMAIN_LANGUAGES
from dr_sad.data.utils import audio_collation, collate_padded, generate_rttm


class TestAudioCollation:
    """Test the audio_collation function."""

    def test_audio_collation_single_sample(self):
        """Test audio collation with a single sample."""
        # Create a single audio sample
        audio_data = np.random.randn(16000)  # 1 second at 16kHz
        batch = [{"waveforms": audio_data}]

        result = audio_collation(batch)

        assert "waveforms" in result
        assert result["waveforms"].shape == (1, 1, 16000)  # (batch, channel, time)
        assert result["waveforms"].dtype == torch.float32

    def test_audio_collation_multiple_samples_same_length(self):
        """Test audio collation with multiple samples of the same length."""
        # Create multiple audio samples of same length
        length = 8000
        batch = [
            {"waveforms": np.random.randn(length)},
            {"waveforms": np.random.randn(length)},
            {"waveforms": np.random.randn(length)},
        ]

        result = audio_collation(batch)

        assert result["waveforms"].shape == (3, 1, length)  # (batch, channel, time)
        assert result["waveforms"].dtype == torch.float32

    def test_audio_collation_multiple_samples_different_lengths(self):
        """
        Test audio collation with multiple samples of different lengths
        (requires padding).
        """
        # Create audio samples of different lengths
        batch = [
            {"waveforms": np.random.randn(8000)},  # 0.5 seconds
            {"waveforms": np.random.randn(16000)},  # 1 second
            {"waveforms": np.random.randn(12000)},  # 0.75 seconds
        ]

        result = audio_collation(batch)

        # Should be padded to the longest length (16000)
        assert result["waveforms"].shape == (3, 1, 16000)
        assert result["waveforms"].dtype == torch.float32

        # Check that padding is applied correctly (padding should be zeros)
        # The first sample should have zeros in the last 8000 positions
        first_sample = result["waveforms"][0, 0, :]
        assert torch.all(first_sample[8000:] == 0)

    def test_audio_collation_empty_batch(self):
        """Test audio collation with empty batch."""
        batch: list[dict[str, Any]] = []

        with pytest.raises((IndexError, ValueError)):
            audio_collation(batch)

    def test_audio_collation_preserves_data(self):
        """Test that audio collation preserves the original audio data."""
        original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        batch = [{"waveforms": original_data}]

        result = audio_collation(batch)

        # Check that the original data is preserved
        torch.testing.assert_close(
            result["waveforms"][0, 0, : len(original_data)],
            torch.tensor(original_data, dtype=torch.float32),
        )


class TestCollatePadded:
    """Test the collate_padded function."""

    def test_collate_padded_complete_batch(self):
        """Test collate_padded with a complete batch containing all expected keys."""
        batch = [
            {
                "waveforms": np.random.randn(8000),
                "domains": 0,
                "annotations": [(0.0, 1.0), (2.0, 3.0)],
                "audio_path": "test1.flac",
            },
            {
                "waveforms": np.random.randn(12000),
                "domains": 1,
                "annotations": [(0.5, 2.5)],
                "audio_path": "test2.flac",
            },
        ]

        result = collate_padded(batch)

        # Check waveforms are properly collated and padded
        assert "waveforms" in result
        assert result["waveforms"].shape == (2, 1, 12000)  # Padded to max length
        assert result["waveforms"].dtype == torch.float32

        # Check domains are converted to tensor
        assert "domains" in result
        assert torch.equal(result["domains"], torch.tensor([0, 1], dtype=torch.float32))

        # Check other fields are kept as lists
        assert "annotations" in result
        assert result["annotations"] == [[(0.0, 1.0), (2.0, 3.0)], [(0.5, 2.5)]]

        assert "audio_path" in result
        assert result["audio_path"] == ["test1.flac", "test2.flac"]

    def test_collate_padded_minimal_batch(self):
        """Test collate_padded with minimal required keys."""
        batch = [
            {"waveforms": np.random.randn(4000), "domains": 0},
            {"waveforms": np.random.randn(6000), "domains": 2},
        ]

        result = collate_padded(batch)

        assert result["waveforms"].shape == (2, 1, 6000)
        assert torch.equal(result["domains"], torch.tensor([0, 2], dtype=torch.float32))

    def test_collate_padded_single_sample(self):
        """Test collate_padded with a single sample."""
        batch = [
            {
                "waveforms": np.random.randn(10000),
                "domains": 1,
                "extra_field": "test_value",
            }
        ]

        result = collate_padded(batch)

        assert result["waveforms"].shape == (1, 1, 10000)
        assert torch.equal(result["domains"], torch.tensor([1], dtype=torch.float32))
        assert result["extra_field"] == ["test_value"]

    def test_collate_padded_preserves_custom_fields(self):
        """Test that collate_padded preserves custom fields as lists."""
        batch = [
            {
                "waveforms": np.random.randn(5000),
                "domains": 0,
                "custom_field1": "value1",
                "custom_field2": [1, 2, 3],
                "custom_field3": {"nested": "dict"},
            },
            {
                "waveforms": np.random.randn(7000),
                "domains": 1,
                "custom_field1": "value2",
                "custom_field2": [4, 5],
                "custom_field3": {"other": "data"},
            },
        ]

        result = collate_padded(batch)

        assert result["custom_field1"] == ["value1", "value2"]
        assert result["custom_field2"] == [[1, 2, 3], [4, 5]]
        assert result["custom_field3"] == [{"nested": "dict"}, {"other": "data"}]


class TestGenerateRttm:
    """Test the generate_rttm function."""

    def test_generate_rttm_basic(self):
        """Test basic RTTM generation."""
        data = {
            "timestamps_start": [1.0, 3.5, 7.2],
            "timestamps_end": [2.5, 5.0, 9.1],
            "speakers": ["spk1", "spk2", "spk1"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            rttm_dir = Path(temp_dir)
            file_id = "test_file"

            generate_rttm(data, rttm_dir, file_id)

            rttm_file = rttm_dir / f"{file_id}.rttm"
            assert rttm_file.exists()

            with open(rttm_file) as f:
                lines = f.readlines()

            assert len(lines) == 3

            # Check first line format
            expected_line1 = (
                "SPEAKER test_file 1 1.000 1.500 <NA> <NA> spk1 <NA> <NA>\n"
            )
            assert lines[0] == expected_line1

            # Check second line format
            expected_line2 = (
                "SPEAKER test_file 1 3.500 1.500 <NA> <NA> spk2 <NA> <NA>\n"
            )
            assert lines[1] == expected_line2

            # Check third line format
            expected_line3 = (
                "SPEAKER test_file 1 7.200 1.900 <NA> <NA> spk1 <NA> <NA>\n"
            )
            assert lines[2] == expected_line3

    def test_generate_rttm_with_none_speakers(self):
        """Test RTTM generation filtering out 'None' speakers."""
        data = {
            "timestamps_start": [1.0, 3.0, 5.0, 7.0],
            "timestamps_end": [2.0, 4.0, 6.0, 8.0],
            "speakers": ["spk1", "None", "spk2", "None"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            rttm_dir = Path(temp_dir)
            file_id = "test_file"

            generate_rttm(data, rttm_dir, file_id)

            rttm_file = rttm_dir / f"{file_id}.rttm"
            with open(rttm_file) as f:
                lines = f.readlines()

            # Should only have 2 lines (None speakers filtered out)
            assert len(lines) == 2
            assert "spk1" in lines[0]
            assert "spk2" in lines[1]
            assert "None" not in "".join(lines)

    def test_generate_rttm_custom_channel(self):
        """Test RTTM generation with custom channel ID."""
        data = {
            "timestamps_start": [0.0],
            "timestamps_end": [1.0],
            "speakers": ["speaker1"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            rttm_dir = Path(temp_dir)
            file_id = "test_file"
            channel_id = 2

            generate_rttm(data, rttm_dir, file_id, channel_id=channel_id)

            rttm_file = rttm_dir / f"{file_id}.rttm"
            with open(rttm_file) as f:
                content = f.read()

            # Should contain the custom channel ID
            assert f"test_file {channel_id}" in content

    def test_generate_rttm_empty_data(self):
        """Test RTTM generation with empty data."""
        data: dict[str, list[Any]] = {
            "timestamps_start": [],
            "timestamps_end": [],
            "speakers": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            rttm_dir = Path(temp_dir)
            file_id = "test_file"

            generate_rttm(data, rttm_dir, file_id)

            rttm_file = rttm_dir / f"{file_id}.rttm"
            assert rttm_file.exists()

            with open(rttm_file) as f:
                content = f.read()

            # File should be empty
            assert content == ""

    def test_generate_rttm_precision(self):
        """Test that RTTM generation maintains proper precision for timestamps."""
        data = {
            "timestamps_start": [1.23456],
            "timestamps_end": [2.98765],
            "speakers": ["test_speaker"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            rttm_dir = Path(temp_dir)
            file_id = "test_file"

            generate_rttm(data, rttm_dir, file_id)

            rttm_file = rttm_dir / f"{file_id}.rttm"
            with open(rttm_file) as f:
                content = f.read()

            # Check that timestamps are formatted to 3 decimal places
            assert "1.235" in content  # start time
            assert "1.753" in content  # duration (2.98765 - 1.23456 = 1.75309)


class TestDomainLanguages:
    """Test the DOMAIN_LANGUAGES constant."""

    def test_domain_languages_structure(self):
        """Test that DOMAIN_LANGUAGES has the expected structure."""
        assert isinstance(DOMAIN_LANGUAGES, dict)

        # Check expected language codes
        expected_languages = {"eng", "deu", "spa", "jpn", "zho"}
        assert set(DOMAIN_LANGUAGES.keys()) == expected_languages

        # Check that values are unique integers
        values = list(DOMAIN_LANGUAGES.values())
        assert len(values) == len(set(values))  # All unique
        assert all(isinstance(v, int) for v in values)
        assert set(values) == {0, 1, 2, 3, 4}

    def test_domain_languages_mapping(self):
        """Test specific language mappings."""
        assert DOMAIN_LANGUAGES["eng"] == 0
        assert DOMAIN_LANGUAGES["deu"] == 1
        assert DOMAIN_LANGUAGES["spa"] == 2
        assert DOMAIN_LANGUAGES["jpn"] == 3
        assert DOMAIN_LANGUAGES["zho"] == 4
