"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysis import evaluate_file, inverse_weightings_by_domain
from dr_sad.evaluating import SpeechDetectionEvaluator
from dr_sad.utils import get_experiment_name

# Collar duration in seconds and detection threshold for evaluation metrics
# We can adjust these later if needed
COLLAR_SECONDS = 0.25
DETECTION_THRESHOLD = 0.5

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def run_analysis(
    prediction_path: Path,
    model_metadata: dict[str, float | int],
    data_name: str,
    results_filepath: Path,
    use_collar: bool = False,
    inverse_weightings: bool = True,
) -> None:
    """
    Run frame level analysis on predictions

    Args:
        prediction_path (Path): Path to the file containing predictions
        model_metadata (dict[str, float | int]): Dictionary containing model frame rate
            and other metadata
        data_name (str): Name of the dataset being analyzed
        results_filepath (Path): Path to save YAML file with aggregated results
        use_collar (bool): Whether to use collar frames in the analysis
        inverse_weightings (bool): Whether to use inverse weightings based on domain
            representation when calculating mean results across files.
    """
    # Load prediction file
    predictions = load_file(prediction_path)

    split_name = prediction_path.stem
    print(f"Loaded predictions from: {prediction_path}")
    print(f"Number of files: {len(predictions)}")

    # Set up evaluator
    if use_collar:
        collar_frames = int(COLLAR_SECONDS * model_metadata["frame_rate_hz"])
        split_name += "_with_collar"
    else:
        collar_frames = 0

    evaluator = SpeechDetectionEvaluator(
        collar_frames=collar_frames,
        detection_threshold=DETECTION_THRESHOLD,
    )

    all_results = {}

    data_dir = MAIN_DIR / "data" / data_name

    # Analyze all files
    for file_id in tqdm(list(predictions.keys()), desc=f"Analysing {split_name} files"):
        numpy_predictions = predictions[file_id].numpy().squeeze()
        evaluation_metrics = evaluate_file(
            data_dir,
            file_id,
            numpy_predictions,
            model_metadata,
            evaluator,
            output_dir=None,
            plot_figures=False,
        )
        all_results[file_id] = evaluation_metrics.to_dict()

    data_tbl_path = data_dir / "sources.tbl"

    mean_results = {}
    if not all_results:
        err_msg = f"No results to analyse for {split_name} - check predictions."
        raise ValueError(err_msg)

    metric_names = next(iter(all_results.values())).keys()
    if inverse_weightings:
        weightings = inverse_weightings_by_domain(
            list(predictions.keys()), data_tbl_path, data_name=data_name
        )
        # Get metric names from first item in dictionary
        for metric_name in metric_names:
            # Build list of (value, weight) pairs for this metric
            values = []
            file_weightings = []
            for file_id, metrics in all_results.items():
                values.append(metrics[metric_name])
                file_weightings.append(weightings[file_id])

            mean_results[metric_name] = np.average(
                values,
                weights=file_weightings,
            ).item()
    else:
        for metric_name in metric_names:
            mean_results[metric_name] = np.mean(
                [metrics[metric_name] for metrics in all_results.values()]
            ).item()

    # Load existing results or create new dict
    if results_filepath.exists():
        with open(results_filepath) as f:
            all_split_results = yaml.safe_load(f) or {}
    else:
        all_split_results = {}

    # Add mean results for this split
    all_split_results = all_split_results | {split_name: mean_results}

    # Save updated results to yaml file
    results_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(results_filepath, "w") as f:
        yaml.dump(all_split_results, f)


def main(
    experiment_config_path: str,
    domain: int | None,
    use_collar: bool,
    split_idx: int,
) -> None:
    # load experiment config to get data name
    _, experiment_path = get_experiment_name(experiment_config_path, EXP_CONFIG_DIR)
    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]
    split_name = data_cfg["split_names"][split_idx].strip(".yaml")

    if (
        data_cfg.get("domain_type") == "exclude_one"
        or data_cfg.get("domain_type") == "single_domain"
    ) and domain is None:
        msg = (
            "Error: When data domain_type is 'exclude_one' or 'single_domain', "
            "--domain argument must be provided."
        )
        raise ValueError(msg)

    output_path = (
        MAIN_DIR / "outputs" / experiment_path.stem / split_name
        if domain is None
        else MAIN_DIR
        / "outputs"
        / experiment_path.stem
        / split_name
        / f"domain_{domain}"
    )
    predictions_paths = list(output_path.glob("saved_predictions/*.safetensors"))
    if not predictions_paths:
        msg = f"No prediction files found in {output_path / 'saved_predictions/'}"
        raise FileNotFoundError(msg)

    experiment_dir = predictions_paths[0].parent.parent

    # Load model metadata
    model_metadata = yaml.safe_load(
        (experiment_dir / "model_metadata.yaml").read_text()
    )

    for prediction_path in predictions_paths:
        run_analysis(
            prediction_path=prediction_path,
            model_metadata=model_metadata,
            data_name=data_name,
            results_filepath=experiment_dir / "frame_metrics.yaml",
            use_collar=use_collar,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze model predictions")
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Path or name to the experiment configuration file.",
    )
    parser.add_argument(
        "split_idx",
        type=int,
        help="Index of the data split to use, read from data config file",
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=None,
        help=(
            "Domain to exclude when domain_type is 'exclude_one', OR"
            " target domain when domain_type is 'single_domain'."
        ),
    )
    parser.add_argument(
        "--use-collar",
        action="store_true",
        help="Whether to use collar frames in the analysis",
    )
    args = parser.parse_args()
    main(
        args.experiment_config,
        args.domain,
        args.use_collar,
        args.split_idx,
    )
