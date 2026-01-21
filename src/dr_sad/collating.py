import re
from collections.abc import Hashable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPECTED_EVALUATIONS = [
    "test",
    "test_with_collar",
    "out_of_domain",
    "out_of_domain_with_collar",
]


def map_domain_indices(df_index: list[int], domain_names: dict[int, str]) -> list[str]:
    """Map numeric domain indices to names, keeping non-numeric indices as-is."""
    return [domain_names.get(idx, str(idx)) for idx in df_index]


def load_domain_metrics(
    experiment_dir: Path, metric_file: str = "frame_metrics.yaml"
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Load metrics from all domain subdirectories.

    Args:
        experiment_dir: Path to experiment directory containing domain_* subdirs
        metric_file: Name of the metric file to load (default: frame_metrics.yaml)

    Returns:
        Dictionary mapping domain indices to their metrics for all splits
    """
    domain_metrics = {}

    # Compile regex once for better performance
    domain_pattern = re.compile(r"domain_(\d+)")

    # Use glob to find all metric files in domain_* directories
    for metric_path in sorted(experiment_dir.glob(f"domain_*/{metric_file}")):
        # Extract domain index from path using regex
        domain_re = domain_pattern.search(metric_path.parent.name)
        if domain_re is None:
            print(f"Warning: Could not parse domain index from path: {metric_path}")
            continue

        domain_idx = int(domain_re.group(1))

        with open(metric_path) as f:
            metrics = yaml.safe_load(f)
            # Extract all relevant splits
            filtered_metrics = {}
            for split_name in EXPECTED_EVALUATIONS:
                if split_name in metrics:
                    filtered_metrics[split_name] = metrics[split_name]

            domain_metrics[domain_idx] = filtered_metrics

    return domain_metrics


def create_metrics_dataframe(
    domain_metrics: dict[int, dict[str, dict[str, float]]], split: str
) -> pd.DataFrame:
    """
    Create a DataFrame from domain metrics for a specific split.

    Args:
        domain_metrics: Dictionary mapping domain indices to their split metrics
        split: The split to extract; must be one of EXPECTED_SPLITS

    Returns:
        DataFrame with domains as rows, metrics as columns, plus mean/std rows
    """
    # Extract metrics for the specified split
    split_data = {}
    for domain_idx, splits in domain_metrics.items():
        if split in splits:
            split_data[domain_idx] = splits[split]

    # Create DataFrame from domain metrics
    df = pd.DataFrame.from_dict(split_data, orient="index")

    # Sort by domain index
    df = df.sort_index()

    # Calculate mean and std
    mean_row = df.mean()
    std_row = df.std(ddof=0)

    # Add mean and std as new rows
    df.loc["mean"] = mean_row
    df.loc["std"] = std_row

    return df


def iter_leaves(
    obj: Any, path: tuple[Hashable, ...] = ()
) -> Iterator[tuple[tuple[Hashable, ...], Any]]:
    """
    Yield (path, value) for each non-dict leaf in a nested mapping.

    Args:
        obj: The object to traverse. If it is a dict, it is traversed recursively.
        path: Current path of keys to the object being visited.

    Yields:
        Tuples of (path, value), where `path` is a tuple of keys leading to `value`.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, (*path, k))
    else:
        yield path, obj


def set_in_tree(
    tree: dict[Hashable, Any], path: Iterable[Hashable], value: Any
) -> None:
    """
    Set a value in a nested dict, creating intermediate dicts as needed.

    This is equivalent to:
      tree[path[0]][path[1]]...[path[-1]] = value

    Args:
        tree: Root dictionary to mutate.
        path: Sequence of keys describing the nested location.
        value: Value to assign at the target path.
    """
    path_tuple = tuple(path)
    if len(path_tuple) == 0:
        msg = "Path must contain at least one key."
        raise ValueError(msg)

    cur: dict[Hashable, Any] = tree
    for k in path_tuple[:-1]:
        cur = cur.setdefault(k, {})
    cur[path_tuple[-1]] = value


def pivot_top_keys_to_leaves(data: dict[Hashable, Any]) -> dict[Hashable, Any]:
    """
    Pivot a nested mapping so top-level keys become leaf-level keys.

    Transforms:
        data[top][...path...] = leaf
    into:
        out[...path...][top] = leaf

    Args:
        data: Mapping of top-level keys to nested mappings.

    Returns:
        A new nested mapping with the top-level keys moved to the leaves.
    """
    out: dict[Hashable, Any] = {}
    for top_key, subtree in data.items():
        for path, leaf_value in iter_leaves(subtree):
            set_in_tree(out, (*path, top_key), leaf_value)
    return out


def add_mean_std_to_tree(
    data: dict[Hashable, Any],
) -> dict[Hashable, Any]:
    """
    Add 'mean' and 'std' entries to the innermost dictionaries of a nested mapping.

    Args:
        data: Nested mapping where innermost values are dictionaries of numeric metrics.

    Returns:
        New nested mapping with 'mean' and 'std' added to innermost dictionaries.
    """
    new_data: dict[Hashable, Any] = {}
    if all(isinstance(v, int | float) for v in data.values()):
        # Innermost dictionary with numeric values
        mean_value = float(np.mean(list(data.values())))
        std_value = float(np.std(list(data.values())))
        new_data = {**data, "mean": mean_value, "std": std_value}

    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = add_mean_std_to_tree(value)
        else:
            new_data[key] = value

    return new_data


def collate_submetrics(
    results_dir: Path | str,
    folder_pattern: str,
    metric_file: str,
):
    """
    Collate metrics across subdirectories matching a pattern into nested dictionaries.

    Args:
        results_dir (Path | str): Path containing subdirectories matching the pattern.
        folder_pattern (str): Pattern to identify subdirectories (e.g., 'domain_*').
        metric_file: Metric filename to read from each subdirectory.

    Returns:
        Nested dictionary keyed by split -> metric -> subdirectory/mean/std.
    """
    if not isinstance(results_dir, Path):
        results_path = Path(results_dir)
    else:
        results_path = results_dir

    if not results_path.is_dir():
        msg = (
            f"Results directory does not exist: {results_path}. Current working"
            f" directory: {Path.cwd()}"
        )
        raise NotADirectoryError(msg)

    metric_paths = sorted(results_path.glob(f"{folder_pattern}/{metric_file}"))

    if len(metric_paths) == 0:
        msg = (
            f"No metric files named '{metric_file}' found in any "
            f"{folder_pattern} subdirectories of {results_path}"
        )
        raise FileNotFoundError(msg)
    if len(metric_paths) == 1:
        print(
            f"Warning: Only one metric file named '{metric_file}' found in "
            f"{results_path}. Did you mean to use collate_metrics instead?"
        )

    loaded_metrics: dict[Hashable, Any] = {}
    this_name: str = ""
    this_metrics: dict[str, Any] = {}
    for metric_path in metric_paths:
        this_name = metric_path.parent.name
        with open(metric_path) as f:
            this_metrics = yaml.safe_load(f)

        loaded_metrics[this_name] = this_metrics

    pivoted_metrics = pivot_top_keys_to_leaves(loaded_metrics)
    return add_mean_std_to_tree(pivoted_metrics)
