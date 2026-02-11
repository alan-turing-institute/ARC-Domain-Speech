from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysis import generate_precision_recall_curve_data
from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.plotting import plot_general_pr_curve, plot_pr_curves, set_plot_style
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"

N_THRESHOLDS = 50


def main(
    experiment_config: str,
) -> None:
    """
    Main function to generate precision-recall curve for model predictions.

    Args:
        experiment_config (str): Path or name to the experiment configuration file.
    """
    set_plot_style()
    # Load predictions and ground truth based on experiment_config
    experiment_name, experiment_cfg_path = get_experiment_name(
        experiment_config, EXP_CONFIG_DIR
    )

    figure_save_path = MAIN_DIR / "outputs" / experiment_name / "figures"
    figure_save_path.mkdir(parents=True, exist_ok=True)

    with open(experiment_cfg_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]
    tbl_path = MAIN_DIR / "data" / data_name / "sources.tbl"
    split_names: list[str] = data_cfg["split_names"]

    domain_name_idx_map = DOMAIN_SETTINGS[data_name]["domains_idx"]
    domain_idx_name_map = {idx: name for name, idx in domain_name_idx_map.items()}

    # Find all domain directories
    base_experiment_output = (
        MAIN_DIR / "outputs" / experiment_name / Path(split_names[0]).stem
    )
    domain_dirs = list(base_experiment_output.glob("domain_*"))
    domains = [int(d.name.replace("domain_", "")) for d in domain_dirs]

    # Initialize dictionary to store all results across domains
    all_results: dict[str, dict[str, dict[str, list[float]]]] = {}

    #  create plot array for each domain and eval split
    num_domains = len(domains)
    ncols = 2
    nrows = int(np.ceil(num_domains / ncols)) if num_domains > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Loop over each domain
    for domain in tqdm(sorted(domains)):
        experiment_output_pattern = (
            MAIN_DIR
            / "outputs"
            / experiment_name
            / split_names[0].rstrip(".yaml")
            / f"domain_{domain}"
        )
        eval_split_names = [
            path.stem
            for path in list(
                experiment_output_pattern.glob("saved_predictions/*.safetensors")
            )
        ]

        if "validation" in eval_split_names:
            eval_split_names.remove("validation")
        if "train" in eval_split_names:
            eval_split_names.remove("train")

        results_dict: dict[str, dict[str, list[float]]] = {
            eval_split: {"precision": [], "recall": []}
            for eval_split in eval_split_names
        }

        model_metadata = yaml.safe_load(
            (experiment_output_pattern / "model_metadata.yaml").read_text()
        )
        for eval_split in eval_split_names:
            for split_name in split_names:
                prediction_path = (
                    MAIN_DIR
                    / "outputs"
                    / experiment_name
                    / split_name.rstrip(".yaml")
                    / f"domain_{domain}"
                    / f"saved_predictions/{eval_split}.safetensors"
                )
                predictions = load_file(prediction_path)

                precision, recall, inverse_weightings = (
                    generate_precision_recall_curve_data(
                        N_THRESHOLDS,
                        predictions,
                        model_metadata,
                        MAIN_DIR / "data" / data_name,
                        data_name,
                        tbl_path,
                    )
                )

                masked_precision = np.ma.masked_invalid(precision)
                masked_recall = np.ma.masked_invalid(recall)

                # Save precision-recall data
                results_dict[eval_split]["precision"].append(
                    np.ma.average(masked_precision, axis=1, weights=inverse_weightings)
                )
                results_dict[eval_split]["recall"].append(
                    np.ma.average(masked_recall, axis=1, weights=inverse_weightings)
                )

        # stack values
        stacked_results = {
            eval_split: {
                "precision": np.array(results_dict[eval_split]["precision"]),
                "recall": np.array(results_dict[eval_split]["recall"]),
            }
            for eval_split in eval_split_names
        }

        # Store results for this domain
        for eval_split, results in stacked_results.items():
            if f"domain_{domain}" not in all_results:
                all_results[f"domain_{domain}"] = {}
            all_results[f"domain_{domain}"][eval_split] = {
                "precision": results["precision"].tolist(),
                "recall": results["recall"].tolist(),
            }

        # save precision-recall curves to matplotlib figure
        for index, eval_split in enumerate(eval_split_names):
            if eval_split == "out_of_domain" or eval_split == "test_single":
                plot_label = domain_idx_name_map[domain].replace("_", " ").capitalize()
            elif eval_split == "test_all_except":
                plot_label = (
                    f"All except {domain_idx_name_map[domain].replace('_', ' ')}"
                )
            else:
                plot_label = eval_split.replace("_", " ").capitalize()

            axes[domain] = plot_pr_curves(
                stacked_results[eval_split],
                plot_label,
                axes[domain],
                colour_index=index,
            )

        axes[domain].set_xlabel("Recall")
        axes[domain].set_ylabel("Precision")
        axes[domain].legend(loc="lower left")
        if domain in domain_idx_name_map:
            axes[domain].set_title(
                f"{domain_idx_name_map[domain].replace('_', ' ').title()}"
            )
        else:
            axes[domain].set_title(f"Domain {domain}")

    fig.tight_layout()
    fig.savefig(figure_save_path / "precision_recall_curves_by_domain.pdf")
    plt.close(fig)

    # Create general precision-recall curve across all domains
    mean_curves = plot_general_pr_curve(
        all_results, figure_save_path, plotting_function="all_curves"
    )
    all_results["mean"] = mean_curves
    # Save all results to files
    results_save_path = figure_save_path.parent / "precision_recall_curve_data.yaml"
    with open(results_save_path, "w") as f:
        yaml.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate precision-recall curves for all domains"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Path or name to the experiment configuration file.",
    )
    args = parser.parse_args()
    main(
        args.experiment_config,
    )
