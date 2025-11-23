"""
Analysis script for model predictions.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysing import evaluate_file
from dr_sad.evaluating import SpeechDetectionEvaluator


def main(prediction_path: Path, plot_figures: bool):
    # Load prediction file
    predictions = load_file(prediction_path)
    model_metadata = yaml.safe_load(
        (prediction_path.parent / "model_metadata.yaml").read_text()
    )

    print(f"Loaded predictions from: {prediction_path}")
    print(f"Number of files: {len(predictions)}")

    # Create output directory in the experiment folder if plotting is enabled
    data_name = prediction_path.stem
    experiment_dir = prediction_path.parent.parent
    if plot_figures:
        analysis_dir = experiment_dir / "analysis_plots" / data_name
    else:
        analysis_dir = None

    # Initialize evaluator
    evaluator = SpeechDetectionEvaluator(
        collar_frames=int(0.25 * model_metadata["frame_rate_hz"]),
        detection_threshold=0.8,
    )

    all_results = {}

    # Analyze all files
    for file_id in tqdm(list(predictions.keys()), desc="Analyzing files"):
        numpy_predictions = predictions[file_id].numpy().squeeze()
        evaluation_metrics = evaluate_file(
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

        # save mean results to json
        mean_results_path = experiment_dir / "mean_metrics" / f"{data_name}.json"
        mean_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mean_results_path, "w") as f:
            json.dump(mean_results, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze model predictions")
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
    main(Path(args.prediction_path), plot_figures=args.plot_figures)
