from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd
import yaml

from dr_sad.collating import (
    check_splits_consistency,
    collate_submetrics,
    remove_unwanted_keys,
)

MAIN_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = MAIN_DIR / "outputs"


@dataclass
class CollateAllMetricsArgs:
    experiment_name: str


def main(experiment_name: str) -> None:
    print(f"Collating all metrics for experiment: {experiment_name}")

    experiment_results_dir = RESULTS_DIR / experiment_name

    splits = glob(str(experiment_results_dir / "split_*"))

    for split_path in splits:
        frame_metrics = collate_submetrics(
            results_dir=split_path,
            folder_pattern="domain_*",
            metric_file="frame_metrics.yaml",
        )
        segment_metrics = collate_submetrics(
            results_dir=split_path,
            folder_pattern="domain_*",
            metric_file="segment_analysis.yaml",
        )

        # Check split consistency
        frame_splits = set(frame_metrics.keys())
        segment_splits = set(segment_metrics["f1_scores"].keys())
        check_splits_consistency(frame_splits, segment_splits)

        # Merge f1_scores into frame_metrics
        for split_name in frame_metrics:
            frame_metrics[split_name]["segment_f1_scores"] = segment_metrics[
                "f1_scores"
            ][split_name]

        # Save collated metrics
        out_path = Path(split_path) / "all_metrics.yaml"
        with open(out_path, "w") as out_file:
            yaml.safe_dump(frame_metrics, out_file)

    experiment_metrics = frame_metrics = collate_submetrics(
        results_dir=experiment_results_dir,
        folder_pattern="split_*",
        metric_file="all_metrics.yaml",
    )
    # Save everything(!)
    out_path = experiment_results_dir / "full_experiment_results.yaml"
    with open(out_path, "w") as out_file:
        yaml.safe_dump(experiment_metrics, out_file)

    # EXAMPLE CSV OUTPUT
    csv_value = remove_unwanted_keys(experiment_metrics, [None, "der", "mean", "mean"])
    df = pd.DataFrame.from_dict(csv_value, orient="index", columns=["der"])
    df.to_csv(experiment_results_dir / "full_experiment_result.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Collate all metrics across domains and splits")
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment to collate metrics for.",
    )
    args: CollateAllMetricsArgs = parser.parse_args()
    main(args.experiment_name)
