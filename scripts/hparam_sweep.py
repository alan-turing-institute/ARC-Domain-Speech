from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dr_sad.plotting import (
    get_hparam_sweep_results_from_all_metrics,
    plot_baseline,
    plot_hparam_sweep_points_error_bars,
    set_plot_style,
)


def main() -> None:
    set_plot_style()

    models = {
        "adversarial_lambda": "Adversarial @ Linear",
        "adversarial_lstm_lambda": "Adversarial @ LSTM",
        "irm_lambda": "IRM",
        "vrex_lambda": "V-REx",
    }

    results: dict[str, dict[str, list[float]]] = {
        model: {"lambdas": [], "means": [], "stds": []} for model in models
    }

    max_DER = 0.0
    min_DER = float("inf")

    for model in models:
        experiment_names = glob(f"outputs/cyclicLR_hparam_sweep_{model}*")

        for experiment_name in experiment_names:
            lambda_value = experiment_name.split("_")[-1].removeprefix("lambda")
            lambda_value = lambda_value.replace("p", ".")

            experiment_results_file = f"{experiment_name}/all_metrics.yaml"

            means, stds = get_hparam_sweep_results_from_all_metrics(
                experiment_results_file
            )
            results[model]["lambdas"].append(float(lambda_value))
            results[model]["means"].append(means)
            results[model]["stds"].append(stds)
            max_DER = max(max_DER, max(means) + max(stds))
            min_DER = min(min_DER, min(means) - max(stds))

    n_rows = 2
    n_cols = len(models) // n_rows + int(len(models) % n_rows > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 10))
    axes = np.atleast_1d(axes).flatten()

    max_DER = ((max_DER // 5) + 1) * 5
    min_DER = ((min_DER // 5) - 1) * 5

    for i, model in enumerate(models):
        lambdas = plot_hparam_sweep_points_error_bars(
            axes[i], results[model], label=None, color=f"C{i}"
        )

        baseline_means, baseline_stds = get_hparam_sweep_results_from_all_metrics(
            "outputs/cyclicLR_baseline_domain/all_metrics.yaml"
        )

        baseline_result_dict = {
            "lambdas": np.array([min(lambdas), max(lambdas)]),
            "means": baseline_means,
            "stds": baseline_stds,
        }
        plot_baseline(
            axes[i],
            baseline_result_dict,
            color="dimgrey",
            mean_linewidth=1.0,
            std_linewidth=0.5,
        )

        axes[i].set_title(models[model])
        axes[i].set_xscale("symlog", linthresh=sorted(lambdas)[1])
        axes[i].set_ylim(min_DER, max_DER)

        if len(axes[i].get_legend_handles_labels()[0]) > 0:
            axes[i].add_artist(axes[i].legend(loc="upper left", title="Model"))

        axes[i].set_xlabel("$\\lambda$")
        axes[i].set_ylabel("DER (%)")

    # Create legend elements to denote the different data splits
    split_legend_elements = [
        Line2D(
            [0],
            [0],
            color="dimgrey",
            lw=1,
            label="Held-out domain",
            marker="o",
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color="dimgrey",
            lw=1,
            linestyle="--",
            marker="x",
            label="Test",
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=split_legend_elements,
        title="Data Split",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncols=len(split_legend_elements),
    )

    fig.suptitle("Hparam Sweep Results", fontsize=16)
    fig.savefig(
        "outputs/figures/CyclicLR_hparam_sweep.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
