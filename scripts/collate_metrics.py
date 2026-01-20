import argparse
from pathlib import Path

import pandas as pd

from dr_sad.collating import (
    EXPECTED_EVALUATIONS,
    create_metrics_dataframe,
    load_domain_metrics,
    map_domain_indices,
)
from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.utils import get_experiment_name

FRAME_METRIC_FILE = "frame_metrics.yaml"
MAIN_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = MAIN_DIR / "outputs"


def main(
    experiment_name: str,
    expected_evaluations: list[str],
    dataset_name: str | None = None,
) -> None:
    exp_name, _ = get_experiment_name(
        exp_name_arg=experiment_name,
        exp_config_dir=MAIN_DIR / "configs" / "experiment",
    )
    experiment_dir = OUTPUTS_DIR / exp_name
    if not experiment_dir.exists():
        err_msg = f"Experiment directory not found: {experiment_dir}"
        raise FileNotFoundError(err_msg)

    print(f"Collating metrics from: {experiment_dir}")

    # Prepare domain name mapping if requested
    domain_names = None
    if dataset_name is not None:
        if dataset_name not in DOMAIN_SETTINGS:
            err_msg = f"Dataset '{dataset_name}' not found in DOMAIN_SETTINGS"
            raise ValueError(err_msg)
        # Reverse mapping: idx -> name
        domains_idx = DOMAIN_SETTINGS[dataset_name]["domains_idx"]
        domain_names = {v: k for k, v in domains_idx.items()}

    # Process frame_metrics.yaml
    print(f"\nProcessing {FRAME_METRIC_FILE}...")

    domain_metrics = load_domain_metrics(
        experiment_dir, FRAME_METRIC_FILE, expected_evaluations
    )

    # Create separate DataFrames for each split
    aggregated_means = {}

    for split in expected_evaluations:
        df = create_metrics_dataframe(domain_metrics, split)

        if df.empty:
            print(f"No data found for split '{split}', skipping")
            continue

        # If mapping is available, add domain name column and set as index
        if domain_names is not None:
            # Only relabel numeric indices, not 'mean'/'std'
            df["domain"] = map_domain_indices(df.index, domain_names)
            df = df.set_index("domain")
        else:
            df.index.name = "domain"

        # Determine output filename
        output_name = f"frame_metrics_{split}.csv"
        output_path = experiment_dir / output_name

        df.to_csv(output_path, float_format="%.4f")
        print(f"Saved split metrics for '{split}': {output_path}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        # Store mean row for aggregated table
        aggregated_means[split] = df.loc["mean"]

    # Create aggregated table with means across all splits
    if aggregated_means:
        aggregated_df = pd.DataFrame.from_dict(aggregated_means, orient="index")
        aggregated_df.index.name = "split"
        aggregated_path = experiment_dir / "frame_metrics_aggregated.csv"
        aggregated_df.to_csv(aggregated_path, float_format="%.4f")
        print(f"\nSaved aggregated means: {aggregated_path}")
        print(
            f"Shape: {aggregated_df.shape[0]} rows x {aggregated_df.shape[1]} columns"
        )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment directory name (e.g. callhome_domain_10.yaml)",
    )
    parser.add_argument(
        "--expected-evaluations",
        type=str,
        nargs="+",
        default=EXPECTED_EVALUATIONS,
        help=(
            "List of expected evaluation splits to process, defaults to "
            "EXPECTED_EVALUATIONS in dr_sad.collating.py"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name for domain mapping (optional)",
    )
    args = parser.parse_args()
    experiment_name = args.experiment_name
    dataset_name = args.dataset_name
    expected_evaluations = args.expected_evaluations
    main(experiment_name, expected_evaluations, dataset_name)
