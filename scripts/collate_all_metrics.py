from argparse import ArgumentParser
from pathlib import Path

import yaml

from dr_sad.collating import (
    check_splits_consistency,
    collate_submetrics,
    metrics_to_table,
)
from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent


def main(experiment_config_path: str) -> None:
    print(f"Collating all metrics for experiment: {experiment_config_path}")

    experiment_name, experiment_path = get_experiment_name(
        experiment_config_path, MAIN_DIR / "configs" / "experiment"
    )
    experiment_results_dir = MAIN_DIR / "outputs" / experiment_name

    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = MAIN_DIR / "configs" / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]

    domain_names = {v: k for k, v in DOMAIN_SETTINGS[data_name]["domains_idx"].items()}

    splits = sorted(experiment_results_dir.glob("split_*"))

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

    # Save all the experiment metrics to the top file
    out_path = experiment_results_dir / "all_metrics.yaml"
    with open(out_path, "w") as out_file:
        yaml.safe_dump(experiment_metrics, out_file)

    der_df = metrics_to_table(
        all_metrics=experiment_metrics,
        metric_name="der",
        domain_names=domain_names,
    )
    der_df.to_csv(experiment_results_dir / "der_table.csv")
    print("DER RESULTS")
    print(der_df)

    frame_f1 = metrics_to_table(
        all_metrics=experiment_metrics,
        metric_name="f1_speech",
        domain_names=domain_names,
    )
    frame_f1.to_csv(experiment_results_dir / "frame_f1_table.csv")
    print("")
    print("FRAME F1 RESULTS")
    print(frame_f1)

    segment_f1 = metrics_to_table(
        all_metrics=experiment_metrics,
        metric_name="segment_f1_scores",
        domain_names=domain_names,
    )
    segment_f1.to_csv(experiment_results_dir / "segment_f1_table.csv")
    print("")
    print("SEGMENT F1 RESULTS")
    print(segment_f1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Collate all metrics across domains and splits")
    parser.add_argument(
        "experiment_config_path",
        type=str,
        help="Path to the experiment configuration file.",
    )
    args = parser.parse_args()
    main(args.experiment_config_path)
