from glob import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dr_sad.plotting import (
    get_hparam_sweep_results_from_csv,
    plot_hparam_sweep_with_error_bands,
    set_plot_style,
)


def main():
    set_plot_style()

    models = {
        "adversarial_lambda": "Adversarial @ Linear",
        "adversarial_lstm_lambda": "Adversarial @ LSTM",
        "irm_lambda": "IRM",
        "vrex_lambda": "V-REx",
    }

    results = {model: {"lambdas": [], "means": [], "stds": []} for model in models}

    for model in models:
        experiment_names = glob(f"outputs/hpsweep_dihard_synthetic_{model}*")

        for experiment_name in experiment_names:
            lambda_value = experiment_name.split("_")[-1].removeprefix("lambda")
            lambda_value = lambda_value.replace("p", ".")

            experiment_results_file = f"{experiment_name}/der_table.csv"

            _, means, stds = get_hparam_sweep_results_from_csv(experiment_results_file)
            results[model]["lambdas"].append(float(lambda_value))
            results[model]["means"].append(means)
            results[model]["stds"].append(stds)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        lambdas = plot_hparam_sweep_with_error_bands(
            ax, results[model], label=models[model], color=f"C{i}"
        )

    baseline_results = get_hparam_sweep_results_from_csv(
        "outputs/dihard_synthetic_domain_30/der_table.csv"
    )

    basline_result_dict = {
        "lambdas": [min(lambdas), max(lambdas)],
        "means": [baseline_results[1], baseline_results[1]],
        "stds": [baseline_results[2], baseline_results[2]],
    }

    plot_hparam_sweep_with_error_bands(
        ax,
        basline_result_dict,
        label="Baseline",
        color="black",
        plot_with_error_bands=False,
    )

    ax.set_xscale("log")
    ax.set_ylim(20, 45)

    ax.add_artist(ax.legend(loc="upper left", title="Model"))

    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("DER (%)")
    ax.set_title("Hparam Sweep Results")

    # Create legend elements to denote the different data splits
    legend_elements = [
        Line2D([0], [0], color="dimgrey", lw=1, label="Held-out domain"),
        Line2D([0], [0], color="dimgrey", lw=1, linestyle="--", label="Test"),
    ]

    # Add a second legend
    ax.add_artist(
        ax.legend(
            handles=legend_elements,
            title="Data",
            loc="upper left",
            bbox_to_anchor=(0.24, 1.00),
        )
    )

    fig.savefig(
        "outputs/figures/hparam_sweep_results.pdf", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
