import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, SymmetricalLogLocator
from matplotlib.transforms import blended_transform_factory

from dr_sad.plotting import (
    get_hparam_sweep_results_from_all_metrics,
    plot_hparam_sweep_points_error_bars,
    plot_single_point,
    set_plot_style,
)

FIG_SIZE = (4, 12)

Y_LIM_OVERRIDE = (20, 40)


def label_erm_tick(ax: plt.Axes, lambdas: np.ndarray) -> None:
    ax.figure.canvas.draw()
    ticks = ax.get_xticks()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    baseline_lambda = -1 * sorted(lambdas)[1]
    closest_idx = int(np.argmin(np.abs(ticks - baseline_lambda)))
    labels[closest_idx] = "ERM"
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.figure.canvas.draw()
    ax.get_xticklabels()[closest_idx].set_ha("center")


def main(filepath_pattern: str) -> None:
    set_plot_style()

    models = {
        "adversarial_lstm_lambda": "AvLSTM",
        "adversarial_lambda": "AvHead",
        "irm_lambda": "IRM",
        "vrex_lambda": "V-REx",
    }
    # models = {
    #     "hparam_sweep_adversarial_domain_gen_lambda": "Linear Multi-class",
    #     "hparam_sweep_adversarial_domain_gen_binary_lambda": "Linear Binary",
    #     "hparam_sweep_adversarial_lstm_domain_gen_lambda": "LSTM Multi-class",
    #     "hparam_sweep_adversarial_lstm_domain_gen_binary_lambda": "LSTM Binary",
    # }

    results: dict[str, dict[str, list[float]]] = {
        model: {"lambdas": [], "means": [], "stds": []} for model in models
    }

    max_DER = 0.0
    min_DER = float("inf")

    for model in models:
        # experiment_names = glob(f"outputs/{filepath_pattern}_{model}*")
        experiment_names = glob(f"outputs/{filepath_pattern}_hparam_sweep_{model}*")

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

    n_rows = 4
    n_cols = len(models) // n_rows + int(len(models) % n_rows > 0)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(FIG_SIZE[0] * n_cols, FIG_SIZE[1])
    )
    axes = np.atleast_1d(axes).flatten()

    max_DER = ((max_DER // 5) + 1) * 5
    min_DER = ((min_DER // 5) - 1) * 5

    colours = [1, 0, 2, 3]

    for i, model in enumerate(models):
        lambdas = plot_hparam_sweep_points_error_bars(
            axes[i], results[model], label=None, color=f"C{colours[i]}"
        )

        if len(lambdas) < 2:
            err_msg = (
                f"Warning: Not enough lambda values found for model {model} to plot a "
                f"meaningful curve. Found lambdas: {lambdas}"
            )
            raise ValueError(err_msg)

        baseline_means, baseline_stds = get_hparam_sweep_results_from_all_metrics(
            f"outputs/{filepath_pattern}_baseline_domain/all_metrics.yaml"
        )

        baseline_result_dict = {
            "lambdas": np.array([min(lambdas), max(lambdas)]),
            "means": baseline_means,
            "stds": baseline_stds,
        }
        # plot_baseline(
        #     axes[i],
        #     baseline_result_dict,
        #     color="dimgrey",
        #     mean_linewidth=1.0,
        #     std_linewidth=0.5,
        # )

        axes[i].text(
            0.05,
            0.86,
            models[model],
            fontsize=16,
            transform=axes[i].transAxes,
            ha="left",
            va="top",
        )

        linthresh = sorted(lambdas)[1] * 1.25
        axes[i].set_xscale("symlog", linthresh=linthresh)

        trans = blended_transform_factory(axes[i].transData, axes[i].transAxes)

        d = 0.5
        kwargs_m = {
            "marker": [(-d, -1), (d, 1)],  # rotated for vertical spines
            "markersize": 12,
            "linestyle": "none",
            "color": "k",
            "mec": "k",
            "mew": 1,
            "clip_on": False,
        }

        axes[i].plot([linthresh * 0.47], [0], transform=trans, **kwargs_m)
        axes[i].plot([linthresh * 0.42], [0], transform=trans, **kwargs_m)

        # tick labels for the minor ticks at 2, 3, ..., 9 times the linthresh
        axes[i].xaxis.set_minor_locator(
            SymmetricalLogLocator(
                linthresh=sorted(lambdas)[1],
                base=10,
                subs=np.arange(2, 10),
            )
        )
        axes[i].set_ylim(*Y_LIM_OVERRIDE if Y_LIM_OVERRIDE else (min_DER, max_DER))

        if len(axes[i].get_legend_handles_labels()[0]) > 0:
            axes[i].add_artist(axes[i].legend(loc="upper left", title="Model"))
        # Only label the x-axis for the bottom row and the y-axis for the leftmost
        # column
        if i >= n_rows - 1:
            axes[i].set_xlabel("$\\lambda$")
        if i % n_cols == 0:
            axes[i].set_ylabel("DER (%)")

        _ = plot_single_point(
            axes[i],
            baseline_result_dict,
            color="dimgrey",
            lambda_value=-1 * sorted(lambdas)[1],
        )

        # plot comparison to original paper
        if args.run_lstm and i == 1:
            single_fig, single_ax = plt.subplots(figsize=(6, 4))
            lambdas = plot_hparam_sweep_points_error_bars(
                single_ax,
                results[model],
                label=None,
                color=f"C{i}",
            )

            if len(lambdas) < 2:
                err_msg = (
                    f"Warning: Not enough lambda values found for model {model} to plot"
                    f" a meaningful curve. Found lambdas: {lambdas}"
                )
                raise ValueError(err_msg)

            baseline_means, baseline_stds = get_hparam_sweep_results_from_all_metrics(
                f"outputs/{filepath_pattern}_baseline_domain/all_metrics.yaml"
            )

            baseline_result_dict = {
                "lambdas": np.array([min(lambdas), max(lambdas)]),
                "means": baseline_means,
                "stds": baseline_stds,
            }

            single_ax.text(
                0.05,
                0.86,
                models[model],
                fontsize=16,
                transform=single_ax.transAxes,
                ha="left",
                va="top",
            )

            single_linthresh = sorted(lambdas)[1] * 1.25
            single_ax.set_xscale("symlog", linthresh=single_linthresh)

            single_trans = blended_transform_factory(
                single_ax.transData, single_ax.transAxes
            )
            single_ax.plot(
                [single_linthresh * 0.47], [0], transform=single_trans, **kwargs_m
            )
            single_ax.plot(
                [single_linthresh * 0.42], [0], transform=single_trans, **kwargs_m
            )

            single_ax.xaxis.set_minor_locator(
                SymmetricalLogLocator(
                    linthresh=sorted(lambdas)[1],
                    base=10,
                    subs=np.arange(2, 10),
                )
            )
            single_ax.set_ylim(*Y_LIM_OVERRIDE if Y_LIM_OVERRIDE else (7, 17))

            _ = plot_single_point(
                single_ax,
                baseline_result_dict,
                color="dimgrey",
                lambda_value=-1 * sorted(lambdas)[1],
            )

            split_legend_elements_single = [
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
                    label="In-Domain",
                    alpha=0.7,
                ),
            ]
            single_fig.legend(
                handles=split_legend_elements_single,
                loc="lower right",
                bbox_to_anchor=(0.95, 0.16),
                ncols=1,
            )

            single_ax.set_xlabel("$\\lambda$")
            single_ax.set_ylabel("DER (%)")

            label_erm_tick(single_ax, lambdas)
            single_fig.tight_layout()
            single_fig.savefig(
                f"outputs/figures/{filepath_pattern}_{model}_hparam_sweep_new.png",
                dpi=300,
                bbox_inches="tight",
            )

        label_erm_tick(axes[i], lambdas)

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
            label="In-Domain",
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=split_legend_elements,
        # title="Data Split",
        loc="upper right",
        bbox_to_anchor=(0.95, 0.75),
        ncols=1,
    )
    fig.tight_layout()
    fig.savefig(
        f"outputs/figures/{filepath_pattern}_hparam_sweep_new_single_col.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "filepath_pattern",
        type=str,
        help=(
            "The pattern to match experiment directories for the hparam sweep"
            " (e.g., 'hparam_sweep' to match directories like "
            "'hparam_sweep_adversarial_lambda0p1')"
        ),
    )
    argparser.add_argument(
        "--run-lstm",
        action="store_true",
        help="Set to True to run the LSTM comparison plot",
    )
    args = argparser.parse_args()
    main(args.filepath_pattern)
