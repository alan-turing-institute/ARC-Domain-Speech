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
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = False


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
    min_recall = np.min(results_dict["recall"][:, :-1])  # last point is often NaN/0
    max_recall = np.max(results_dict["recall"][:, :-1])  # last point is often NaN/0
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
            " Must be 'interpolated', 'all_curves', or 'all'."
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

            ax = plot_curves(
                means_over_domains,
                eval_split.replace("_", " ").capitalize(),
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
