import tempfile
from collections.abc import Hashable
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import yaml

from dr_sad.collating import (
    add_mean_std_to_tree,
    collate_submetrics,
    create_metrics_dataframe,
    iter_leaves,
    load_domain_metrics,
    map_domain_indices,
    pivot_top_keys_to_leaves,
    remove_unwanted_keys,
    set_in_tree,
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


class TestIterLeaves:
    def test_iter_leaves_simple(self):
        data = {"a": 1, "b": {"c": 2}}

        result = sorted(iter_leaves(data))

        assert result == [(("a",), 1), (("b", "c"), 2)]

    def test_iter_leaves_nested(self):
        data = {"x": {"y": {"z": 3}}, "w": 4}

        result = sorted(iter_leaves(data))

        assert result == [(("w",), 4), (("x", "y", "z"), 3)]

    def test_iter_leaves_non_dict(self):
        result = list(iter_leaves(5))

        assert result == [((), 5)]


class TestSetInTree:
    def test_set_in_tree_creates_paths(self):
        tree: dict[Hashable, Any] = {}

        set_in_tree(tree, ("a", "b", "c"), 1)

        assert tree == {"a": {"b": {"c": 1}}}

    def test_set_in_tree_overwrites_value(self):
        tree: dict[Hashable, Any] = {"a": {"b": {"c": 1}}}

        set_in_tree(tree, ("a", "b", "c"), 2)

        assert tree["a"]["b"]["c"] == 2

    def test_set_in_tree_empty_path_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            set_in_tree({}, (), 1)


class TestPivotTopKeysToLeaves:
    def test_pivot_top_keys_to_leaves_basic(self):
        data: dict[Hashable, Any] = {
            "domain_0": {"split": {"metric": 0.1}},
            "domain_1": {"split": {"metric": 0.2}},
        }

        result = pivot_top_keys_to_leaves(data)

        assert result == {"split": {"metric": {"domain_0": 0.1, "domain_1": 0.2}}}

    def test_pivot_top_keys_to_leaves_multiple_metrics(self):
        data: dict[Hashable, Any] = {
            "domain_0": {"split": {"m1": 1.0, "m2": 2.0}},
            "domain_1": {"split": {"m1": 3.0, "m2": 4.0}},
        }

        result = pivot_top_keys_to_leaves(data)

        assert result["split"]["m1"] == {"domain_0": 1.0, "domain_1": 3.0}
        assert result["split"]["m2"] == {"domain_0": 2.0, "domain_1": 4.0}


class TestAddMeanStdToTree:
    def test_add_mean_std_single_level(self):
        data: dict[Hashable, Any] = {"metric": {"domain_0": 1.0, "domain_1": 3.0}}

        result = add_mean_std_to_tree(data)

        assert result["metric"]["mean"] == pytest.approx(2.0)
        assert result["metric"]["std"] == pytest.approx(1.0)

    def test_add_mean_std_nested(self):
        data: dict[Hashable, Any] = {
            "split": {"metric": {"domain_0": 2.0, "domain_1": 4.0, "domain_2": 6.0}}
        }

        result = add_mean_std_to_tree(data)

        assert result["split"]["metric"]["mean"] == pytest.approx(4.0)
        assert result["split"]["metric"]["std"] == pytest.approx(1.632993)

    def test_add_mean_std_preserves_values(self):
        data: dict[Hashable, Any] = {"metric": {"domain_0": 1.0, "domain_1": 2.0}}

        result = add_mean_std_to_tree(data)

        assert result["metric"]["domain_0"] == 1.0
        assert result["metric"]["domain_1"] == 2.0

    def test_add_mean_std_multiple_metrics(self):
        data: dict[Hashable, Any] = {
            "split": {
                "m1": {"domain_0": 1.0, "domain_1": 3.0},
                "m2": {"domain_0": 2.0, "domain_1": 4.0},
            }
        }

        result = add_mean_std_to_tree(data)

        assert result["split"]["m1"]["mean"] == pytest.approx(2.0)
        assert result["split"]["m1"]["std"] == pytest.approx(1.0)
        assert result["split"]["m2"]["mean"] == pytest.approx(3.0)
        assert result["split"]["m2"]["std"] == pytest.approx(1.0)

    def test_add_mean_std_mixed_types_no_aggregate(self):
        data: dict[Hashable, Any] = {"metric": {"domain_0": 1.0, "note": "skip"}}

        result = add_mean_std_to_tree(data)

        assert "mean" not in result["metric"]
        assert "std" not in result["metric"]
        assert result["metric"]["domain_0"] == 1.0
        assert result["metric"]["note"] == "skip"


class TestCollateSubmetrics:
    @pytest.fixture()
    def temp_metrics(self, tmp_path):
        domain_0 = tmp_path / "domain_0"
        domain_1 = tmp_path / "domain_1"
        domain_2 = tmp_path / "domain_2"
        domain_0.mkdir()
        domain_1.mkdir()
        domain_2.mkdir()

        metrics_0 = {
            "in_domain_test": {"test_accuracy": 0.9, "test_loss": 0.1},
            "out_of_domain_test": {"test_accuracy": 0.8, "test_loss": 0.2},
        }
        metrics_1 = {
            "in_domain_test": {"test_accuracy": 0.7, "test_loss": 0.3},
            "out_of_domain_test": {"test_accuracy": 0.6, "test_loss": 0.4},
        }
        metrics_2 = {
            "in_domain_test": {"test_accuracy": 0.8, "test_loss": 0.15},
            "out_of_domain_test": {"test_accuracy": 0.65, "test_loss": 0.35},
        }

        with open(domain_0 / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics_0, f)
        with open(domain_1 / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics_1, f)
        with open(domain_2 / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics_2, f)

        expected = {
            "domain_0": metrics_0,
            "domain_1": metrics_1,
            "domain_2": metrics_2,
        }

        return tmp_path, expected

    def test_collate_reads_metrics(self, temp_metrics):
        results_dir, expected = temp_metrics

        result = collate_submetrics(results_dir, "domain_*", "metrics.yaml")

        for split_name, split_metrics in expected["domain_0"].items():
            assert split_name in result
            for metric_name in split_metrics:
                assert metric_name in result[split_name]

    def test_collate_values_match_inputs(self, temp_metrics):
        results_dir, expected = temp_metrics

        result = collate_submetrics(results_dir, "domain_*", "metrics.yaml")

        for domain_name, domain_metrics in expected.items():
            for split_name, split_metrics in domain_metrics.items():
                for metric_name, metric_value in split_metrics.items():
                    assert result[split_name][metric_name][domain_name] == metric_value

    def test_collate_adds_mean_std(self, temp_metrics):
        results_dir, _ = temp_metrics

        result = collate_submetrics(results_dir, "domain_*", "metrics.yaml")

        for split_name in result:
            for _, metric_values in result[split_name].items():
                domain_values = [
                    value
                    for key, value in metric_values.items()
                    if key.startswith("domain_")
                ]
                mean_expected = float(pd.Series(domain_values).mean())
                std_expected = float(pd.Series(domain_values).std(ddof=0))

                assert metric_values["mean"] == pytest.approx(mean_expected)
                assert metric_values["std"] == pytest.approx(std_expected)

    def test_collate_raises_on_missing_metrics(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collate_submetrics(tmp_path, "domain_*", "metrics.yaml")

    def test_collate_warns_single_domain(self, tmp_path, capsys):
        domain_0 = tmp_path / "domain_0"
        domain_0.mkdir()
        with open(domain_0 / "metrics.yaml", "w") as f:
            yaml.safe_dump({"split": {"metric": 1.0}}, f)

        collate_submetrics(tmp_path, "domain_*", "metrics.yaml")

        captured = capsys.readouterr()
        assert "Only one metric file named" in captured.out


class TestRemoveUnwantedKeys:
    def test_all_none_pattern_returns_original(self):
        """Test that all-None pattern returns the original data unchanged."""
        data: dict[Hashable, Any] = {
            "split": {"metric": {"domain_0": 1.0, "domain_1": 2.0}}
        }
        key_pattern: list[str | None] = [None, None, None]

        result = remove_unwanted_keys(data, key_pattern)

        assert result == data

    def test_all_string_pattern_direct_lookup(self):
        """Test that all-string pattern performs direct nested lookup."""
        data: dict[Hashable, Any] = {
            "split": {"metric": {"domain_0": 1.0, "domain_1": 2.0}}
        }
        key_pattern: list[str | None] = ["split", "metric"]

        result = remove_unwanted_keys(data, key_pattern)

        assert result == {None: {"domain_0": 1.0, "domain_1": 2.0}}

    def test_all_string_pattern_missing_keys(self):
        """Test all-string pattern with missing keys returns empty dict under None."""
        data: dict[Hashable, Any] = {"split": {"metric": {"domain_0": 1.0}}}
        key_pattern: list[str | None] = ["split", "missing"]

        with pytest.raises(KeyError, match="not found in data"):
            remove_unwanted_keys(data, key_pattern)

    def test_mixed_pattern_selective_matching(self):
        """Test mixed None/string pattern filters paths selectively."""
        data: dict[Hashable, Any] = {
            "split": {
                "m1": {"domain_0": 1.0, "domain_1": 2.0},
                "m2": {"domain_0": 3.0, "domain_1": 4.0},
            }
        }
        key_pattern: list[str | None] = ["split", "m1", None]

        result = remove_unwanted_keys(data, key_pattern)

        assert result == {"domain_0": 1.0, "domain_1": 2.0}

    def test_mixed_pattern_with_trailing_none(self):
        """Test mixed pattern with trailing None keeps remaining path levels."""
        data: dict[Hashable, Any] = {
            "split": {"metric": {"domain_0": 1.0, "domain_1": 2.0}}
        }
        key_pattern: list[str | None] = ["split", None, None]

        result = remove_unwanted_keys(data, key_pattern)

        assert result == {"metric": {"domain_0": 1.0, "domain_1": 2.0}}

    def test_invalid_pattern__raises_error(self):
        """Test that pattern longer than path raises ValueError."""
        data: dict[Hashable, Any] = {"split": {"metric": 1.0}}
        key_pattern: list[str | None] = ["split", "metric", "extra"]

        with pytest.raises(ValueError, match="Invalid key pattern"):
            remove_unwanted_keys(data, key_pattern)

    def test_pattern_longer_than_path_raises_error(self):
        """Test that pattern longer than path raises ValueError."""
        data: dict[Hashable, Any] = {"split": {"metric": 1.0}}
        key_pattern: list[str | None] = ["split", None, "extra"]

        with pytest.raises(ValueError, match="Key pattern is longer than path"):
            remove_unwanted_keys(data, key_pattern)

    def test_pattern_mismatch_excludes_path(self):
        """Test that non-matching paths are excluded from result."""
        data: dict[Hashable, Any] = {
            "split1": {"metric": 1.0},
            "split2": {"metric": 2.0},
        }
        key_pattern: list[str | None] = ["split1", None]

        result = remove_unwanted_keys(data, key_pattern)

        assert "split2" not in str(result)
        assert result == {"metric": 1.0}

    def test_complex_nested_with_selective_filter(self):
        """Test complex nested structure with selective filtering."""
        data: dict[Hashable, Any] = {
            "train": {
                "accuracy": {"d0": 0.9, "d1": 0.85},
                "loss": {"d0": 0.1, "d1": 0.15},
            },
            "test": {
                "accuracy": {"d0": 0.8, "d1": 0.75},
                "loss": {"d0": 0.2, "d1": 0.25},
            },
        }
        key_pattern: list[str | None] = ["train", None, "d0"]

        result = remove_unwanted_keys(data, key_pattern)

        assert result == {"accuracy": 0.9, "loss": 0.1}
        assert "test" not in str(result)
        assert "d1" not in str(result)
