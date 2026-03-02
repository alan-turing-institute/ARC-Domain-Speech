import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.linewidth"] = 1.6
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["font.size"] = 11
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.grid"] = True


def _interpolate_pr_curves(
    original_recall: np.ndarray,
    original_precision: np.ndarray,
    new_recall: np.ndarray,
) -> np.ndarray:
    """
    Interpolate precision values at new recall points for multiple curves.

    Args:
        original_recall: Array of shape (n_curves, n_points) with original recall values
        original_precision: Array of shape (n_curves, n_points) with original precision
            values
        new_recall: Array of shape (n_new_points,) with new recall values to interpolate
            at

    Returns:
        interpolated_precision: Array of shape (n_curves, n_new_points) with
            interpolated precision
    """
    n_curves, _ = original_recall.shape
    n_new_points = len(new_recall)

    interpolated_precision = np.zeros((n_curves, n_new_points))

    for curve_idx in range(n_curves):
        recall_curve = original_recall[curve_idx]
        precision_curve = original_precision[curve_idx]

        # Remove NaN values
        valid_mask = ~np.isnan(recall_curve) & ~np.isnan(precision_curve)
        recall_clean = recall_curve[valid_mask]
        precision_clean = precision_curve[valid_mask]

        if len(recall_clean) < 2:
            # Not enough points for interpolation, fill with NaN
            interpolated_precision[curve_idx] = np.nan
            continue

        # Sort by recall for interpolation (required by np.interp)
        sorted_indices = np.argsort(recall_clean)
        recall_sorted = recall_clean[sorted_indices]
        precision_sorted = precision_clean[sorted_indices]

        # Interpolate precision to new recall values
        interpolated_precision[curve_idx] = np.interp(
            new_recall, recall_sorted, precision_sorted
        )

    return interpolated_precision


def plot_pr_curves(
    results_dict: dict[str, np.ndarray],
    plot_label: str,
    ax: plt.Axes,
    colour_index: int = 0,
) -> plt.Axes:
    """
    Plot precision-recall curves.

    Args:
        results_dict: Dictionary containing precision and recall arrays for a split
        plot_label: Label for the curve to use in the legend
        ax: Matplotlib Axes object to plot on
        colour_index: Index to determine colour of the plot (for consistency across
            plots)
    Returns:
        ax: Matplotlib Axes object with the plot
    """
    precision_values = results_dict["precision"]
    recall_values = results_dict["recall"]

    # Calculate mean and std ignoring NaN values
    mean_precision = np.nanmean(precision_values, axis=0)
    mean_recall = np.nanmean(recall_values, axis=0)

    # Plot mean curve
    ax.plot(
        mean_recall[:-1],  # last point is often NaN/0
        mean_precision[:-1],
        label=plot_label,
        color=f"C{colour_index}",
    )
    for i in range(precision_values.shape[0]):
        ax.plot(
            recall_values[i][:-1],  # last point is often NaN/0
            precision_values[i][:-1],  # last point is often NaN/0
            color=f"C{colour_index}",
            alpha=0.2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return ax


def plot_interpolated_pr_curve(
    results_dict: dict[str, np.ndarray],
    plot_label: str,
    ax: plt.Axes,
    colour_index: int = 0,
) -> plt.Axes:
    """
    Plot precision-recall curve with variance using fill_between and np.interpolate.

    Args:
        results_dict: Dictionary containing precision and recall arrays for a split
        plot_label: Label for the curve to use in the legend
        ax: Matplotlib Axes object to plot on
        colour_index: Index to determine colour of the plot (for consistency across
            plots)
    Returns:
        ax: Matplotlib Axes object with the plot
    """

    # get range of recall values
    min_recall = np.nanmin(results_dict["recall"][:, :-1])  # last point is often NaN/0
    max_recall = np.nanmax(results_dict["recall"][:, :-1])  # last point is often NaN/0
    recall_values = np.linspace(min_recall, max_recall, 50)

    interpolated_precision = _interpolate_pr_curves(
        results_dict["recall"], results_dict["precision"], recall_values
    )

    # Calculate mean and std ignoring NaN values
    mean_precision = np.nanmean(interpolated_precision, axis=0)
    std_precision = np.nanstd(interpolated_precision, axis=0)

    # Plot mean curve
    ax.plot(
        recall_values[:-1],  # last point is often NaN/0
        mean_precision[:-1],  # last point is often NaN/0
        label=plot_label,
        color=f"C{colour_index}",
    )

    # Plot variance
    ax.fill_between(
        recall_values[:-1],  # last point is often NaN/0
        mean_precision[:-1] - std_precision[:-1],
        mean_precision[:-1] + std_precision[:-1],
        alpha=0.4,
        color=f"C{colour_index}",
    )

    return ax


def plot_general_pr_curve(
    all_results: dict[str, dict[str, dict[str, list[float]]]],
    figure_save_path: Path,
    plotting_function: str = "all_curves",
) -> dict[str, dict[str, list[float]]]:
    """
    Create a general precision-recall curve aggregating results across all domains.

    Args:
        all_results: Dictionary with domain results, structure:
                    {domain_X: {eval_split: {precision: [...], recall: [...]}}}
        figure_save_path: Path to save the figure
    """
    if plotting_function == "interpolated":
        plot_curves = plot_interpolated_pr_curve
    elif plotting_function == "all_curves":
        plot_curves = plot_pr_curves
    else:
        err_msg = (
            f"Invalid plotting function: {plotting_function}."
            " Must be 'interpolated' or 'all_curves'."
        )
        raise ValueError(err_msg)

    # Get eval splits from first domain (they're all the same)
    first_domain = next(iter(all_results.values()))
    eval_splits = list(first_domain.keys())

    # Aggregate precision/recall across all domains for each eval split
    aggregated_results = {}
    for eval_split in eval_splits:
        all_precision = []
        all_recall = []
        # repeats x domains x points
        for _, domain_results in all_results.items():
            if eval_split in domain_results:
                # Convert back to numpy arrays
                precision_array = np.array(domain_results[eval_split]["precision"])
                recall_array = np.array(domain_results[eval_split]["recall"])

                all_precision.append(precision_array)
                all_recall.append(recall_array)

        aggregated_results[eval_split] = {
            "precision": np.stack(all_precision, axis=1),
            "recall": np.stack(all_recall, axis=1),
        }

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_results = {}

    for index, eval_split in enumerate(eval_splits):
        if eval_split in aggregated_results:
            means_over_domains: dict[str, np.ndarray] = {
                "precision": np.nanmean(
                    aggregated_results[eval_split]["precision"], axis=1
                ),
                "recall": np.nanmean(aggregated_results[eval_split]["recall"], axis=1),
            }
            mean_results[eval_split] = {
                key: item.tolist() for key, item in means_over_domains.items()
            }

            if eval_split == "out_of_domain":
                plot_label = "Held-out domain"
            elif eval_split == "test":
                plot_label = "Trained on domains"
            else:
                plot_label = eval_split.replace("_", " ").capitalize()

            ax = plot_curves(
                means_over_domains,
                plot_label,
                ax,
                colour_index=index,
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Across All Domains")
    ax.legend()

    # Save the figure
    fig.savefig(
        figure_save_path / "precision_recall_curve_all_domains.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
    return mean_results


def format_value_with_std(value_str: str, std_digits: str) -> tuple[str, float, float]:
    """
    The rightmost digit in parentheses corresponds to the last decimal place of the
    value, each digit to the left represents the next higher decimal place.

        eg. "1.23(4)" means 1.23 ± 0.04, "1.2(34)" means 1.2 ± 0.34, "1(234)"
        means 1 ± 2.34

    args:
        value_str: The mean value as a string, e.g. "1.23"
        std_digits: The digits representing the standard deviation, e.g. "4"

    """
    decimal_places = len(value_str.split(".")[1]) if "." in value_str else 0
    std_value = int(std_digits) * (10**-decimal_places)
    string_representation = f"{value_str} ± {std_value:.{decimal_places}f}"
    return string_representation, float(value_str), float(std_value)


def parse_parentheses_notation(notation: str) -> tuple[str, float, float]:
    """Parse parentheses notation into ± format."""

    match = re.match(r"([0-9]+\.?[0-9]*)\(([0-9]+)\)", notation)
    if not match:
        err_msg = (
            f"Invalid notation format: {notation}. Expected format is 'mean(std)'."
        )
        raise ValueError(err_msg)

    value_str, std_digits = match.groups()

    return format_value_with_std(value_str, std_digits)


vectorized_parse_csv = np.vectorize(parse_parentheses_notation)


def get_hparam_sweep_results_from_csv(csv_path: str) -> np.ndarray:
    """Read baseline results from a CSV file and parse the parentheses notation."""
    vals = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        dtype=object,
    )[-2, 1:3]
    return vectorized_parse_csv(vals)


def plot_hparam_sweep_with_error_bands(
    axis: plt.Axes,
    data: dict[str, list[float]],
    label: str,
    color: str,
    plot_with_error_bands: bool = True,
):
    """
    Plot hparam sweep results with optional error bands.

    args:
        axis: Matplotlib axis to plot on
        data: Dictionary containing 'lambdas', 'means', and 'stds' lists
        label: Label for the plot
        color: Color for the plot
        plot_with_error_bands: Whether to plot error bands using std values
    returns:
        the unsorted lambda values for potential use in plotting other curves on
        the same axis.
    """

    sorted_indices = np.argsort(data["lambdas"])
    lambdas = np.array(data["lambdas"])[sorted_indices]
    means = np.stack(data["means"])[sorted_indices]
    stds = np.stack(data["stds"])[sorted_indices]

    axis.plot(lambdas, means[:, 0], label=label, color=color)
    if plot_with_error_bands:
        axis.fill_between(
            lambdas,
            means[:, 0] - stds[:, 0],
            means[:, 0] + stds[:, 0],
            alpha=0.2,
            color=color,
        )
    axis.plot(lambdas, means[:, 1], "--", color=color, alpha=0.7)
    if plot_with_error_bands:
        axis.fill_between(
            lambdas,
            means[:, 1] - stds[:, 1],
            means[:, 1] + stds[:, 1],
            alpha=0.1,
            color=color,
        )
    return lambdas
