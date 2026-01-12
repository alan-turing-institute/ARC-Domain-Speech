import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd

from dr_sad.collating import (
    create_metrics_dataframe,
    load_domain_metrics,
    map_domain_indices,
)


class TestMapDomainIndices:
    def test_basic_mapping(self):
        """Test basic domain index to name mapping."""
        df_index = [0, 1, 2]
        domain_names = {0: "domain_a", 1: "domain_b", 2: "domain_c"}

        result = map_domain_indices(df_index, domain_names)

        assert result == ["domain_a", "domain_b", "domain_c"]

    def test_missing_domain_fallback(self):
        """Test fallback to string conversion for missing domain indices."""
        df_index = [0, 1, 99]  # 99 not in domain_names
        domain_names = {0: "domain_a", 1: "domain_b"}

        result = map_domain_indices(df_index, domain_names)

        assert result == ["domain_a", "domain_b", "99"]


class TestLoadDomainMetrics:
    def test_load_metrics_success(self):
        """Test successful loading of domain metrics from YAML files."""
        # Mock YAML content
        mock_yaml_content = {
            "test": {"accuracy": 0.85, "f1": 0.82},
            "test_with_collar": {"accuracy": 0.88, "f1": 0.85},
            "out_of_domain": {"accuracy": 0.75, "f1": 0.72},
            "other_split": {"accuracy": 0.90, "f1": 0.87},  # Should be filtered out
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create domain directories with metric files
            (temp_path / "domain_0").mkdir()
            (temp_path / "domain_1").mkdir()

            # Mock file reading
            with (
                patch("builtins.open", mock_open(read_data="dummy")),
                patch("yaml.safe_load", return_value=mock_yaml_content),
                patch.object(Path, "glob") as mock_glob,
            ):
                # Mock glob to return our test paths
                mock_glob.return_value = [
                    temp_path / "domain_0" / "frame_metrics.yaml",
                    temp_path / "domain_1" / "frame_metrics.yaml",
                ]

                result = load_domain_metrics(temp_path)

                # Check structure
                assert 0 in result
                assert 1 in result

                # Check that only expected splits are included
                for domain_metrics in result.values():
                    assert "test" in domain_metrics
                    assert "test_with_collar" in domain_metrics
                    assert "out_of_domain" in domain_metrics
                    assert "other_split" not in domain_metrics

    def test_load_metrics_invalid_domain_path(self):
        """Test handling of invalid domain directory names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with (
                patch.object(Path, "glob") as mock_glob,
                patch("builtins.print") as mock_print,
            ):
                # Mock glob to return invalid path
                mock_glob.return_value = [
                    temp_path / "invalid_name" / "frame_metrics.yaml"
                ]

                result = load_domain_metrics(temp_path)

                # Should return empty dict and print warning
                assert result == {}
                mock_print.assert_called_once()
                assert "Warning: Could not parse domain index" in str(
                    mock_print.call_args
                )


class TestCreateMetricsDataframe:
    def test_create_dataframe_basic(self):
        """Test basic DataFrame creation with mean and std rows."""
        domain_metrics = {
            0: {
                "test": {"accuracy": 0.85, "f1": 0.82},
                "out_of_domain": {"accuracy": 0.75, "f1": 0.72},
            },
            1: {
                "test": {"accuracy": 0.90, "f1": 0.88},
                "out_of_domain": {"accuracy": 0.80, "f1": 0.78},
            },
        }

        result = create_metrics_dataframe(domain_metrics, "test")

        # Check basic structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)  # 2 domains + mean + std rows, 2 metrics
        assert list(result.columns) == ["accuracy", "f1"]

        # Check domain indices are present
        assert 0 in result.index
        assert 1 in result.index
        assert "mean" in result.index
        assert "std" in result.index

        # Check values for domain 0
        assert result.loc[0, "accuracy"] == 0.85
        assert result.loc[0, "f1"] == 0.82

        # Check mean calculation
        expected_mean_accuracy = (0.85 + 0.90) / 2
        assert result.loc["mean", "accuracy"] == expected_mean_accuracy

    def test_create_dataframe_missing_split(self):
        """Test DataFrame creation when some domains don't have the requested split."""
        domain_metrics = {
            0: {
                "test": {"accuracy": 0.85, "f1": 0.82},
            },
            1: {
                "out_of_domain": {"accuracy": 0.80, "f1": 0.78}  # Missing "test" split
            },
        }

        result = create_metrics_dataframe(domain_metrics, "test")

        # Should only include domain 0 since domain 1 doesn't have "test" split
        assert result.shape == (3, 2)  # 1 domain + mean + std rows, 2 metrics
        assert 0 in result.index
        assert 1 not in result.index
        assert "mean" in result.index
        assert "std" in result.index

        # Mean and std should be based only on domain 0
        assert result.loc["mean", "accuracy"] == 0.85
        assert result.loc["std", "accuracy"] == 0.0  # Only one value, so std is 0
