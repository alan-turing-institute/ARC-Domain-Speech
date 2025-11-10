import numpy as np
import pandas as pd
import pytest

from dr_sad.data import data_fetching


class TestRemoveOverlap:
    def test_no_overlap(self):
        segments = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == segments

    def test_one_overlap(self):
        segments = [(0.0, 1.0), (0.5, 2.0), (3.0, 4.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == [(0.0, 2.0), (3.0, 4.0)]

    def test_multiple_overlaps(self):
        segments = [(0.0, 1.0), (0.5, 2.0), (1.5, 3.0), (4.0, 5.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == [(0.0, 3.0), (4.0, 5.0)]

    def test_touching_segments(self):
        segments = [(0.0, 1.0), (1.0, 2.0), (3.0, 4.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == [(0.0, 2.0), (3.0, 4.0)]

    def test_inside_segment(self):
        segments = [(0.0, 3.0), (1.0, 2.0), (4.0, 5.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == [(0.0, 3.0), (4.0, 5.0)]

    def test_type_output(self):
        segments = [(0.0, 1.0), (0.5, 2.0)]
        merged = data_fetching.remove_overlap(segments)
        assert all(isinstance(seg, tuple) for seg in merged)

    def test_single_segment(self):
        segments = [(0.0, 1.0)]
        merged = data_fetching.remove_overlap(segments)
        assert merged == [(0.0, 1.0)]


class TestFullFilePull:
    def test_full_file_pull(self, test_dataset):
        file_id = "TEST_0001"
        d_idx = 0
        data_dir_loc = str(test_dataset)

        result = data_fetching.full_file_pull(file_id, d_idx, data_dir_loc)

        # Check that the result is a dictionary with the expected structure
        assert isinstance(result, dict)
        assert file_id in result
        assert "waveforms" in result[file_id]
        assert "annotations" in result[file_id]
        assert "domains" in result[file_id]

        # Check that waveforms is a numpy array
        assert isinstance(result[file_id]["waveforms"], np.ndarray)

        # Check that annotations is a list of tuples
        assert isinstance(result[file_id]["annotations"], list)
        assert all(isinstance(seg, tuple) for seg in result[file_id]["annotations"])

        # Check that domains is an integer
        assert isinstance(result[file_id]["domains"], int)
        assert result[file_id]["domains"] == d_idx


class TestLoadData:
    def test_load_test_dataset(self, test_dataset):
        """Test loading the test dataset using the fixture."""
        df = data_fetching.load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Check basic structure
        assert len(df) == 20
        assert list(df.columns) == ["waveforms", "annotations", "domains"]

        # Check that we have the expected domains (cycling through AAA, BBB, CCC)
        domain_counts = df["domains"].value_counts().sort_index()
        # With 20 files cycling through 3 domains:
        # 7 files for AAA (0), 7 for BBB (1), 6 for CCC (2)
        assert domain_counts[0] == 7  # AAA
        assert domain_counts[1] == 7  # BBB
        assert domain_counts[2] == 6  # CCC

        # Check audio properties for first file
        first_waveform = df.iloc[0]["waveforms"]
        assert len(first_waveform) == 16000 * 2.5  # 2.5 seconds at 16kHz

        # Check annotations - should have 2 speech segments
        first_annotations = df.iloc[0]["annotations"]
        assert len(first_annotations) == 2
        # First segment: 0.5-1.0, Second segment: 1.5-2.0
        assert first_annotations[0] == (0.5, 1.0)
        assert first_annotations[1] == (1.5, 2.0)

    def test_loaded_data_includes_file_ids(self, test_dataset):
        """Test that loaded data includes file IDs."""
        df = data_fetching.load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
        )

        # Check that 'file_id' column exists and has correct values
        sources = pd.read_csv(test_dataset / "sources.tbl", index_col=0, sep="\t")
        expected_file_ids = sorted(sources.index.tolist())
        loaded_file_ids = sorted(df.index.tolist())
        assert loaded_file_ids == expected_file_ids

    def test_load_data_with_num_workers(self, test_dataset):
        """Test loading data with multiple workers."""
        df = data_fetching.load_data(
            data_choice=None,
            data_set_path=test_dataset,
            domain_column="domain",
            domains_idx={"AAA": 0, "BBB": 1, "CCC": 2},
            num_workers=4,
        )

        # Check basic structure
        assert len(df) == 20
        assert list(df.columns) == ["waveforms", "annotations", "domains"]

        indexes = sorted(
            pd.read_csv(
                test_dataset / "sources.tbl", index_col=0, sep="\t"
            ).index.tolist()
        )

        assert df.index.to_list() == indexes

    def test_missing_data_set_path_raises(self):
        """If data_choice is None and data_set_path is not provided."""

        with pytest.raises(ValueError, match="Either data_choice or data_set_path"):
            data_fetching.load_data(data_choice=None, data_set_path=None)

    def test_missing_domain_column_raises(self, test_dataset):
        """If domain_column is None when data_choice is None, raise ValueError."""

        with pytest.raises(ValueError, match="domain_column must be provided"):
            data_fetching.load_data(
                data_choice=None,
                data_set_path=test_dataset,
                domain_column=None,
                domains_idx={"AAA": 0},
            )

    def test_missing_domains_idx_raises(self, test_dataset):
        """If domains_idx is None when data_choice is None, raise ValueError."""
        with pytest.raises(ValueError, match="domains_idx must be provided"):
            data_fetching.load_data(
                data_choice=None,
                data_set_path=test_dataset,
                domain_column="domain",
                domains_idx=None,
            )

    def test_unknown_data_choice_raises(self):
        """If an unknown data_choice is provided, raise ValueError."""
        with pytest.raises(ValueError, match="Unknown data_choice"):
            data_fetching.load_data(data_choice="unknown_dataset")

    def test_load_invalid_path_raises(self):
        """If an invalid data_set_path is provided, raise ValueError."""
        with pytest.raises(ValueError, match=r"Data directory .* does not exist."):
            data_fetching.load_data(
                data_choice=None,
                data_set_path="non_existent_path",
                domain_column="domain",
                domains_idx={"AAA": 0},
            )
