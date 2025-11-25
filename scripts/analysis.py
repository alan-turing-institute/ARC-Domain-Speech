"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysing import evaluate_file
from dr_sad.evaluating import SpeechDetectionEvaluator
from dr_sad.utils import get_experiment_name

# Collar duration in seconds and detection threshold for evaluation metrics
# We can adjust these later if needed
COLLAR_SECONDS = 0.25
DETECTION_THRESHOLD = 0.8
MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def main(prediction_path: Path, experiment_config_path: str, plot_figures: bool):
    # Load prediction file
    predictions = load_file(prediction_path)
    model_metadata = yaml.safe_load(
        (prediction_path.parent.parent / "model_metadata.yaml").read_text()
    )
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
        analysis_dir = experiment_dir / "analysis_plots" / split_name
    else:
        analysis_dir = None

    # Initialize evaluator
    collar_frames = int(COLLAR_SECONDS * model_metadata["frame_rate_hz"])

    evaluator = SpeechDetectionEvaluator(
        collar_frames=collar_frames,
        detection_threshold=DETECTION_THRESHOLD,
    )

    all_results = {}

    # Analyze all files
    for file_id in tqdm(list(predictions.keys()), desc="Analyzing files"):
        numpy_predictions = predictions[file_id].numpy().squeeze()
        evaluation_metrics = evaluate_file(
            data_name,
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
        results_filepath = experiment_dir / "frame_metrics.yaml"
        if results_filepath.exists():
            with open(results_filepath) as f:
                all_split_results = yaml.safe_load(f) or {}
        else:
            all_split_results = {}

        # Add mean results for this split
        all_split_results[split_name] = mean_results

        # Save updated results to yaml file
        results_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(results_filepath, "w") as f:
            yaml.dump(all_split_results, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze model predictions")
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
