import re
from pathlib import Path

import pandas as pd
import yaml

EXPECTED_SPLITS = [
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
            for split_name in EXPECTED_SPLITS:
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
