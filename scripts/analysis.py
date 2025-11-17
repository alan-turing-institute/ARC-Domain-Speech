"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysing import analyse_file


def main(prediction_path: Path):
    # Load prediction file
    predictions = load_file(prediction_path)

    print(f"Loaded predictions from: {prediction_path}")
    print(f"Number of files: {len(predictions)}")

    # Create output directory in the experiment folder
    experiment_dir = prediction_path.parent.parent  # outputs/test_domain/domain_0
    analysis_dir = experiment_dir / "analysis_plots"
    data_name = prediction_path.stem
    print(f"Saving plots to: {analysis_dir}")

    # Analyze the first few files
    for file_id in tqdm(list(predictions.keys()), desc="Analyzing files"):
        tensor = predictions[file_id]
        analyse_file(file_id, tensor, output_dir=Path(analysis_dir / data_name))


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
