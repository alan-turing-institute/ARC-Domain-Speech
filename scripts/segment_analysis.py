"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file

from dr_sad.segment import SegmentEvaluator
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent

TOLERANCE_SECONDS = 0.25
# Evaluation parameters
THRESHOLD_ON = 0.6
THRESHOLD_OFF = 0.4
MIN_DURATION_OFF = 0.2
MIN_DURATION_ON = 0.3


def load_annotations(
    file_id: str,
    data_path: Path,
) -> list[tuple[float, float]]:
    """Load RTTM annotations.

    Args:
        file_id: Identifier for the audio file.
        data_path: Directory containing the RTTM files.

    Returns:
        List of speech segments as tuples of (start_time, end_time).
    """
    # Load RTTM annotations
    rttm_path = data_path / "rttm" / f"{file_id}.rttm"
    speech_segments = []

    with open(rttm_path) as f:
        for line in f:
            if line.strip() and line.startswith("SPEAKER"):
                parts = line.strip().split()
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speech_segments.append((start_time, end_time))

    return speech_segments


def evaluate_set(
    prediction_data_path: Path,
    data_name: str,
) -> tuple[str, float]:
    """Evaluate the data in the given path.

    Args:
        prediction_data_path (Path): Path to the prediction data file.
        data_name (str): Name of the dataset used.

    Returns:
        set_name (str): Name of the evaluated set.
        f1_score (float): F1 score of the predictions.
    """

    data_dir = MAIN_DIR / "data" / data_name

    model_metadata = yaml.safe_load(
        (prediction_data_path.parent.parent / "model_metadata.yaml").read_text()
    )
    time_start = model_metadata["frame_center_start"] / model_metadata["sample_rate"]
    time_step = model_metadata["frame_center_step"] / model_metadata["sample_rate"]

    # Load predictions and convert to numpy arrays
    st_predictions = load_file(prediction_data_path)
    predictions = {
        file_id: pred_tensor.flatten().numpy()
        for file_id, pred_tensor in st_predictions.items()
    }

    print(f"Loaded predictions from: {prediction_data_path.resolve()}")
    print(f"Number of files: {len(predictions)}")

    references = {}
    for file_id in predictions:
        speech_segments = load_annotations(
            file_id=file_id,
            data_path=data_dir,
        )
        references[file_id] = speech_segments

    # Evaluate predictions
    seg_evaluator = SegmentEvaluator(
        prediction_set=predictions,
        reference_set=references,
        time_start=time_start,
        time_step=time_step,
        tolerance=TOLERANCE_SECONDS,
        threshold_on=THRESHOLD_ON,
        threshold_off=THRESHOLD_OFF,
        min_duration_off=MIN_DURATION_OFF,
        min_duration_on=MIN_DURATION_ON,
    )

    f1_score = seg_evaluator.f1_score()
    set_name = prediction_data_path.stem
    print(f"Set: {set_name}, F1 Score: {f1_score:.4f}")
    return set_name, f1_score


def main(experiment_config_path: str, exclude_domain: int | None):
    # load experiment config to get data name
    _, experiment_path = get_experiment_name(
        experiment_config_path, MAIN_DIR / "configs" / "experiments"
    )
    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = MAIN_DIR / "configs" / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]

    output_path = (
        MAIN_DIR / "outputs" / experiment_path.stem
        if exclude_domain is None
        else MAIN_DIR / "outputs" / experiment_path.stem / f"domain_{exclude_domain}"
    )
    predictions_paths = list(output_path.glob("saved_predictions/*.safetensors"))

    f1_scores = {}

    for prediction_path in predictions_paths:
        set_name, set_f1 = evaluate_set(
            prediction_data_path=prediction_path,
            data_name=data_name,
        )
        f1_scores[set_name] = set_f1

    output_file_path = output_path / "segment_analysis.yaml"
    with open(output_file_path, "w") as file:
        yaml.dump(f1_scores, file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze model predictions")
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Path to or name of the experiment configuration file.",
    )
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude when domain_type is 'exclude_one'",
    )
    args = parser.parse_args()
    main(
        args.experiment_config,
        exclude_domain=args.exclude_domain,
    )
