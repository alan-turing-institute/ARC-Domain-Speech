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


def validate_cached_results_structure(
    all_results: dict[str, dict[str, dict[str, list[float]]]],
    domain_key: str,
    eval_split_names: list[str],
) -> None:
    """
    Validate that the cached precision-recall results contain the expected structure
    for the given domain and evaluation splits.

    Args:
        all_results (dict): The loaded cached results from YAML.
        domain_key (str): The key corresponding to the current domain
            (e.g., "domain_0").
        eval_split_names (list[str]): The list of expected evaluation split names.

    Raises:
        ValueError: If the cached results are missing expected keys or have an
        incompatible structure.
    """
    # Validate that the cached results contain the expected structure
    missing_reasons = []

    if domain_key not in all_results:
        missing_reasons.append(f"missing domain key '{domain_key}' in cached results")
    else:
        missing_eval_splits = []
        for eval_split in eval_split_names:
            if eval_split not in all_results[domain_key]:
                missing_eval_splits.append(
                    f"missing eval split '{eval_split}' under '{domain_key}'"
                )
            else:
                split_entry = all_results[domain_key][eval_split]
                if not isinstance(split_entry, dict):
                    missing_eval_splits.append(  # type: ignore[unreachable]
                        f"eval split '{eval_split}' under '{domain_key}' is not a dict"
                    )
                if "precision" not in split_entry or "recall" not in split_entry:
                    missing_eval_splits.append(
                        f"eval split '{eval_split}' under '{domain_key}' contains keys"
                        f" which are not 'precision' and 'recall'. Found keys: "
                        f"{list(split_entry.keys())}."
                    )

        if missing_eval_splits:
            missing_reasons.extend(missing_eval_splits)

    if missing_reasons:
        raise ValueError(
            "Cached precision-recall results are incompatible with the current "
            "configuration. Detected the following issues:\n - "
            + "\n - ".join(missing_reasons)
            + "\nPlease delete or regenerate the cached YAML file and rerun."
        )


def main(
    experiment_config: str,
    use_existing_results: bool,
) -> None:
    """
    Main function to generate precision-recall curve for model predictions.

    Args:
        experiment_config (str): Path or name to the experiment configuration file.
        use_existing_results (bool): Whether to use existing precision-recall curve
            data.
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
    if len(domain_dirs) == 0:
        err_msg = (
            f"No domain directories found in expected path {base_experiment_output}."
        )
        raise FileNotFoundError(err_msg)

    domains = [int(d.name.replace("domain_", "")) for d in domain_dirs]

    # Initialize dictionary to store all results across domains
    all_results: dict[str, dict[str, dict[str, list[float]]]] = {}

    #  create plot array for each domain and eval split
    num_domains = len(domains)
    nrows = 2
    ncols = int(np.ceil(num_domains / nrows)) if num_domains > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 7))
    axes = np.atleast_1d(axes).flatten()

    if len(axes) > num_domains:
        for extra_ax in axes[num_domains:]:
            extra_ax.axis("off")

    loaded_results = False
    # check results already exist
    if (
        Path(figure_save_path.parent / "precision_recall_curve_data.yaml").is_file()
        and use_existing_results
    ):
        with open(
            Path(figure_save_path.parent / "precision_recall_curve_data.yaml")
        ) as file:
            all_results = yaml.safe_load(file)
        loaded_results = True

    # Loop over each domain
    for domain in tqdm(sorted(domains)):
        experiment_output_pattern = (
            MAIN_DIR
            / "outputs"
            / experiment_name
            / split_names[0].rstrip(".yaml")
            / f"domain_{domain}"
        )

        # Determine eval split names from cached data or filesystem
        if loaded_results:
            domain_key = f"domain_{domain}"
            if domain_key in all_results:
                eval_split_names = list(all_results[domain_key].keys())
            else:
                eval_split_names = []
        else:
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

        if not loaded_results:
            model_metadata = yaml.safe_load(
                (experiment_output_pattern / "model_metadata.yaml").read_text()
            )
            for eval_split in eval_split_names:
                for split_name in split_names:
                    prediction_path = (
                        MAIN_DIR
                        / "outputs"
                        / experiment_name
                        / split_name.removesuffix(".yaml")
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
                        np.ma.average(
                            masked_precision, axis=1, weights=inverse_weightings
                        )
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
        else:
            domain_key = f"domain_{domain}"

            # ensure the loaded results have the expected structure for this domain
            validate_cached_results_structure(all_results, domain_key, eval_split_names)

            stacked_results = {
                eval_split: {
                    "precision": np.array(
                        all_results[domain_key][eval_split]["precision"]
                    ),
                    "recall": np.array(all_results[domain_key][eval_split]["recall"]),
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
            if eval_split == "test_single":
                plot_label = f"{plot_label} only"

            axes[domain] = plot_pr_curves(
                stacked_results[eval_split],
                plot_label,
                axes[domain],
                colour_index=index,
            )

        axes[domain].set_xlabel("Recall", fontsize=14)
        axes[domain].set_ylabel("Precision", fontsize=14)
        axes[domain].legend(loc="lower left", fontsize=12)
        if domain in domain_idx_name_map:
            axes[domain].set_title(
                f"{domain_idx_name_map[domain].replace('_', ' ').title()}", fontsize=16
            )
        else:
            axes[domain].set_title(f"Domain {domain}")

        axes[domain].set_xlim(0.5, 1)
        axes[domain].set_ylim(0.5, 1)

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
    parser.add_argument(
        "--use-existing-results",
        action="store_true",
        help="Whether to use existing precision-recall curve data if it exists, instead"
        " of regenerating it from predictions. If set, the script will look for a file "
        "named 'precision_recall_curve_data.yaml'.",
        default=False,
    )
    args = parser.parse_args()
    main(
        args.experiment_config,
        use_existing_results=args.use_existing_results,
    )
