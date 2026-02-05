from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plot_precision_recall_curve(
    results_dict: dict[str, np.ndarray],
    eval_split: str,
    ax: plt.Axes,
) -> plt.Axes:
    """
    Plot precision-recall curve with variance using fill_between.

    Args:
        results_dict: Dictionary containing precision and recall arrays for a split
        eval_split: The evaluation split name (key for results_dict)
        ax: Matplotlib Axes object to plot on
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
        recall_values, mean_precision, label=eval_split.capitalize().replace("_", " ")
    )

    # Plot variance
    ax.fill_between(
        recall_values,
        mean_precision - std_precision,
        mean_precision + std_precision,
        alpha=0.4,
    )

    return ax


def plot_general_precision_recall_curve(
    all_results: dict[str, dict[str, dict[str, list[float]]]],
    figure_save_path: Path,
) -> None:
    """
    Create a general precision-recall curve aggregating results across all domains.

    Args:
        all_results: Dictionary with domain results, structure:
                    {domain_X: {eval_split: {precision: [...], recall: [...]}}}
        figure_save_path: Path to save the figure
    """
    # Get eval splits from first domain (they're all the same)
    first_domain = next(iter(all_results.values()))
    eval_splits = list(first_domain.keys())

    # Aggregate precision/recall across all domains for each eval split
    aggregated_results = {}
    for eval_split in eval_splits:
        all_precision = []
        all_recall = []

        for _, domain_results in all_results.items():
            if eval_split in domain_results:
                # Convert back to numpy arrays
                precision_array = np.array(domain_results[eval_split]["precision"])
                recall_array = np.array(domain_results[eval_split]["recall"])

                # Add all curves from this domain
                if len(precision_array.shape) == 1:
                    # Single curve
                    all_precision.append(precision_array)
                    all_recall.append(recall_array)
                else:
                    # Multiple curves
                    for i in range(precision_array.shape[0]):
                        all_precision.append(precision_array[i])
                        all_recall.append(recall_array[i])

        if all_precision:
            aggregated_results[eval_split] = {
                "precision": np.array(all_precision),
                "recall": np.array(all_recall),
            }

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for eval_split in eval_splits:
        if eval_split in aggregated_results:
            ax = plot_precision_recall_curve(
                aggregated_results[eval_split], eval_split, ax
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Across All Domains")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save the figure
    fig.savefig(
        figure_save_path / "precision_recall_curve_all_domains.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
