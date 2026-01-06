"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file
from torch import Tensor
from tqdm import tqdm

from dr_sad.analysis import evaluate_file
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
    predictions: dict[str, Tensor],
    evaluator: SpeechDetectionEvaluator,
    model_metadata: dict[str, float | int],
    data_name: str,
    analysis_dir: Path | None,
    split_name: str,
    results_filepath: Path,
    plot_figures: bool,
):
    """
    Run analysis on predictions given an evaluator object

    Args:
        predictions: Dictionary mapping file IDs to prediction tensors
        evaluator: SpeechDetectionEvaluator instance for computing metrics
        model_metadata: Dictionary containing model frame rate and other metadata
        data_name: Name of the dataset being analysed
        analysis_dir: Directory to save analysis plots (None if no plotting)
        split_name: Name of the data split being analysed (e.g., 'test', 'validation')
        results_filepath: Path to save YAML file with aggregated results
        plot_figures: Whether to generate and save analysis plots
    """
    all_results = {}

    data_dir = MAIN_DIR / "data" / data_name

    # Analyze all files
    for file_id in tqdm(list(predictions.keys()), desc=f"Analyzing {split_name} files"):
        numpy_predictions = predictions[file_id].numpy().squeeze()
        evaluation_metrics = evaluate_file(
            data_dir,
            file_id,
            numpy_predictions,
            model_metadata,
            output_dir=analysis_dir,
            evaluator=evaluator,
            plot_figures=plot_figures,
        )
        all_results[file_id] = evaluation_metrics.to_dict()

    # Filter out None results and calculate means
    valid_results: list[dict[str, float]] = list(all_results.values())

    if len(valid_results) > 0:
        mean_results = {}
        # Get metric names from first valid result
        for metric in valid_results[0]:
            mean_results[metric] = sum(
                result[metric] for result in valid_results
            ) / len(valid_results)

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


def main(prediction_path: Path, experiment_config_path: str, plot_figures: bool):
    # Load prediction file
    predictions = load_file(prediction_path)

    # load experiment config to get data name
    _, experiment_path = get_experiment_name(experiment_config_path, EXP_CONFIG_DIR)
    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]

    print(f"Loaded predictions from: {prediction_path}")
    print(f"Number of files: {len(predictions)}")

    # Create output directory in the experiment folder if plotting is enabled
    split_name = prediction_path.stem
    experiment_dir = prediction_path.parent.parent
    if plot_figures:
        analysis_dir_with_collar = (
            experiment_dir / "analysis_plots" / f"{split_name}_with_collar"
        )
        analysis_dir_no_collar = experiment_dir / "analysis_plots" / split_name
    else:
        analysis_dir_with_collar = None
        analysis_dir_no_collar = None

    # Load model metadata
    model_metadata = yaml.safe_load(
        (prediction_path.parent.parent / "model_metadata.yaml").read_text()
    )

    # Initialize evaluator with collar and threshold
    print("Running analysis with collar frames")
    collar_frames = int(COLLAR_SECONDS * model_metadata["frame_rate_hz"])
    evaluator = SpeechDetectionEvaluator(
        collar_frames=collar_frames,
        detection_threshold=DETECTION_THRESHOLD,
    )
    run_analysis(
        predictions=predictions,
        evaluator=evaluator,
        model_metadata=model_metadata,
        data_name=data_name,
        analysis_dir=analysis_dir_with_collar,
        split_name=f"{split_name}_with_collar",
        results_filepath=experiment_dir / "frame_metrics.yaml",
        plot_figures=plot_figures,
    )

    # evaluation with no collar
    print("Running analysis with no collar frames")
    evaluator = SpeechDetectionEvaluator(
        collar_frames=0,
        detection_threshold=DETECTION_THRESHOLD,
    )
    run_analysis(
        predictions=predictions,
        evaluator=evaluator,
        model_metadata=model_metadata,
        data_name=data_name,
        analysis_dir=analysis_dir_no_collar,
        split_name=split_name,
        results_filepath=experiment_dir / "frame_metrics.yaml",
        plot_figures=plot_figures,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyse model predictions")
    parser.add_argument(
        "--experiment-config",
        type=str,
        required=True,
        help="Path or name to the experiment configuration file.",
    )
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Path to the prediction safetensors file",
    )
    parser.add_argument(
        "--plot-figures",
        action="store_true",
        help="Whether to generate and save analysis plots for each file",
    )
    args = parser.parse_args()
    main(
        Path(args.prediction_path),
        args.experiment_config,
        plot_figures=args.plot_figures,
    )
