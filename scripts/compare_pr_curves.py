from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dr_sad.plotting import plot_pr_curves, set_plot_style
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def main(
    exp_configs: list[str],
    evaluation_splits: list[str],
):
    set_plot_style()
    # check the experiments have the PR curve data, and load it if so
    curve_data = {}
    for exp_config in exp_configs:
        experiment_name, _ = get_experiment_name(exp_config, EXP_CONFIG_DIR)
        pr_curve_data_pth = (
            MAIN_DIR / "outputs" / experiment_name / "precision_recall_curve_data.yaml"
        )
        if not pr_curve_data_pth.exists():
            msg = (
                f"PR curve data not found for experiment config {exp_config} at "
                f"expected path {pr_curve_data_pth}. Please run "
                "scripts/precision_recall_curve.py for this experiment config before "
                "running this script."
            )
            raise FileNotFoundError(msg)
        with open(pr_curve_data_pth) as f:
            curve_data[experiment_name] = yaml.safe_load(f)

    figure_save_path = MAIN_DIR / "outputs" / "figures" / "general_pr_curve.pdf"
    figure_save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for experiment_index, (eval_split, (experiment_name, data)) in enumerate(
        zip(evaluation_splits, curve_data.items(), strict=True)
    ):
        experiment_curve_data = {
            "precision": np.array(data["mean"][eval_split]["precision"]),
            "recall": np.array(data["mean"][eval_split]["recall"]),
        }
        ax = plot_pr_curves(
            experiment_curve_data,
            experiment_name,
            ax,
            experiment_index,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figure_save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate precision-recall curves for all domains"
    )
    parser.add_argument(
        "--exp_configs",
        nargs="+",
        type=str,
        required=True,
        help="Path to experiment config file (relative to configs/experiment/)",
    )
    parser.add_argument(
        "--evaluation_splits",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Name of evaluation split to use for each experiment config, eg. "
            "'test', 'out_of_domain', 'test_single' etc."
        ),
    )
    args = parser.parse_args()
    main(args.exp_configs, args.evaluation_splits)
