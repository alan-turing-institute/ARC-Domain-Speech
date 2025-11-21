"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysing import analyse_file
from dr_sad.evaluating import SpeechDetectionEvaluator


def main(prediction_path: Path):
    # Load prediction file
    predictions = load_file(prediction_path)
    model_metadata = yaml.safe_load(
        (prediction_path.parent / "model_metadata.yaml").read_text()
    )

    print(f"Loaded predictions from: {prediction_path}")
    print(f"Number of files: {len(predictions)}")

    # Create output directory in the experiment folder
    experiment_dir = prediction_path.parent.parent
    analysis_dir = experiment_dir / "analysis_plots"
    data_name = prediction_path.stem
    print(f"Saving plots to: {analysis_dir}")

    # Initialize evaluator
    evaluator = SpeechDetectionEvaluator(
        collar_frames=int(0.25 * model_metadata["frame_rate_hz"]),
        detection_threshold=0.8,
    )

    # Analyze all files
    for file_id in tqdm(list(predictions.keys()), desc="Analyzing files"):
        numpy_predictions = predictions[file_id].numpy().squeeze()
        analyse_file(
            file_id,
            numpy_predictions,
            model_metadata,
            output_dir=Path(analysis_dir / data_name),
            evaluator=evaluator,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze model predictions")
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Path to the prediction safetensors file",
    )
    args = parser.parse_args()
    main(Path(args.prediction_path))
