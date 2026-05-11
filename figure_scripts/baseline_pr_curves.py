from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MultipleLocator

from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.plotting import plot_pr_curves, set_plot_style
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"

CURVE_CONFIGS = [
    ("all", "test_single", "Trained on all"),
    ("domain", "out_of_domain", "Trained on all\nexcept domain"),
    ("single", "test_single", "Trained on domain\nonly"),
]

NOISE_DOMAIN_MAP = {
    "restaurant": "Babble",
    "socio_field": "Reverb",
    "clinical": "Musan Noise",
    "meeting": "Volume Changes",
    "webvideo": "Downsample",
}


def load_pr_data(exp_name: str) -> dict:
    path = MAIN_DIR / "outputs" / exp_name / "precision_recall_curve_data.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_domain_idx_name_map(exp_cfg_path: Path) -> dict[int, str]:
    with open(exp_cfg_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_path = CONFIG_DIR / "data" / exp_config["data_config"]
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    domain_name_idx_map = DOMAIN_SETTINGS[data_cfg["name"]]["domains_idx"]
    return {idx: name for name, idx in domain_name_idx_map.items()}


def main(
    all_config: str,
    single_config: str,
    domain_config: str,
) -> None:
    set_plot_style()

    all_exp_name, all_cfg_path = get_experiment_name(all_config, EXP_CONFIG_DIR)
    single_exp_name, _ = get_experiment_name(single_config, EXP_CONFIG_DIR)
    domain_exp_name, _ = get_experiment_name(domain_config, EXP_CONFIG_DIR)

    exp_data = {
        "all": load_pr_data(all_exp_name),
        "single": load_pr_data(single_exp_name),
        "domain": load_pr_data(domain_exp_name),
    }

    domain_idx_name_map = get_domain_idx_name_map(all_cfg_path)
    domains = sorted(
        int(k.replace("domain_", ""))
        for k in exp_data["all"]
        if k.startswith("domain_")
    )

    figure_save_path = MAIN_DIR / "outputs" / all_exp_name / "figures"
    figure_save_path.mkdir(parents=True, exist_ok=True)

    num_domains = len(domains)
    nrows = 2
    ncols = int(np.ceil(num_domains / nrows)) if num_domains > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 7))
    axes = np.atleast_1d(axes).flatten()

    for extra_ax in axes[num_domains:]:
        extra_ax.axis("off")

    first_col_domains = {d for d in domains if d % ncols == 0}
    bottom_row_domains = {
        max(d for d in domains if d % ncols == col) for col in range(ncols)
    }

    for domain in domains:
        domain_key = f"domain_{domain}"
        ax = axes[domain]

        for colour_index, (config_key, split_name, label) in enumerate(CURVE_CONFIGS):
            split_data = exp_data[config_key][domain_key][split_name]
            results = {
                "precision": np.array(split_data["precision"]),
                "recall": np.array(split_data["recall"]),
            }
            ax = plot_pr_curves(results, label, ax, colour_index=colour_index)

        ax.get_legend().remove()
        if domain in first_col_domains:
            ax.set_ylabel("Precision", fontsize=14)
        if domain in bottom_row_domains:
            ax.set_xlabel("Recall" if domain in bottom_row_domains else "", fontsize=14)

        title = (
            domain_idx_name_map[domain].replace("_", " ").title()
            if domain in domain_idx_name_map
            else f"Domain {domain}"
        )
        if title == "Broadcast Interview":
            title = "Broadcast\nInterview"
        ax.text(
            0.05,
            0.05,
            f"{title} \n + {NOISE_DOMAIN_MAP[domain_idx_name_map[domain]]}",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        ax.set_xlim(0.5, 1.01)
        ax.set_ylim(0.5, 1.01)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes[domains[0]].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.85, 0.3),
        fontsize=14,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(
        figure_save_path / "baseline_pr_curves_by_domain.pdf", bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Plot baseline precision-recall curves across three training"
        " configs"
    )
    parser.add_argument(
        "--all",
        required=True,
        type=str,
        help="Path or name to the all experiment configuration file.",
    )
    parser.add_argument(
        "--single",
        type=str,
        required=True,
        help="Path or name to the single experiment configuration file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Path or name to the domain experiment configuration file.",
    )
    args = parser.parse_args()
    main(
        args.all,
        args.single,
        args.domain,
    )
