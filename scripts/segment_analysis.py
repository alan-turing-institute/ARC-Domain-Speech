"""
Analysis script for model predictions.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from safetensors.torch import load_file

from dr_sad.analysis import load_annotations
from dr_sad.segment import SegmentEvaluator
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent

TOLERANCE_SECONDS = 0.25
# Starting evaluation parameters
SPEECH_THRESHOLD = 0.5
GAP_THRESHOLD = 0.2
MIN_DURATION_OFF = 0.2
MIN_DURATION_ON = 0.3


def get_best_parameters(
    validation_data_path: Path,
    data_name: str,
) -> dict[str, float | None]:
    """Get the best parameters for the given validation data.

    Args:
        validation_data_path (Path): Path to the validation data file.
        data_name (str): Name of the dataset used.

    Returns:
        best_params (dict): Dictionary containing the best parameter values.
    """

    data_dir = MAIN_DIR / "data" / data_name

    model_metadata = yaml.safe_load(
        (validation_data_path.parent.parent / "model_metadata.yaml").read_text()
    )
    time_start = model_metadata["frame_center_start"] / model_metadata["sample_rate"]
    time_step = model_metadata["frame_center_step"] / model_metadata["sample_rate"]

    # Load predictions and convert to numpy arrays
    st_predictions = load_file(validation_data_path)
    predictions = {
        file_id: pred_tensor.flatten().numpy()
        for file_id, pred_tensor in st_predictions.items()
    }

    references = {}
    for file_id in predictions:
        speech_segments = load_annotations(file_id, data_dir)
        references[file_id] = speech_segments

    # Evaluate predictions
    seg_evaluator = SegmentEvaluator(
        prediction_set=predictions,
        reference_set=references,
        time_start=time_start,
        time_step=time_step,
        tolerance=TOLERANCE_SECONDS,
        speech_threshold=SPEECH_THRESHOLD,
        gap_threshold=GAP_THRESHOLD,
        min_duration_off=MIN_DURATION_OFF,
        min_duration_on=MIN_DURATION_ON,
    )

    seg_evaluator.optimise_diff_evol(
        speech_threshold=True,
        gap_threshold=True,
        min_duration_off=True,
        min_duration_on=True,
    )

    return seg_evaluator.get_parameters()


def evaluate_set(
    prediction_data_path: Path,
    data_name: str,
    speech_threshold: float = SPEECH_THRESHOLD,
    gap_threshold: float | None = GAP_THRESHOLD,
    min_duration_off: float | None = MIN_DURATION_OFF,
    min_duration_on: float | None = MIN_DURATION_ON,
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
        speech_segments = load_annotations(file_id, data_dir)
        references[file_id] = speech_segments

    # Evaluate predictions
    seg_evaluator = SegmentEvaluator(
        prediction_set=predictions,
        reference_set=references,
        time_start=time_start,
        time_step=time_step,
        tolerance=TOLERANCE_SECONDS,
        speech_threshold=speech_threshold,
        gap_threshold=gap_threshold,
        min_duration_off=min_duration_off,
        min_duration_on=min_duration_on,
    )

    f1_score = seg_evaluator.f1_score()
    set_name = prediction_data_path.stem
    print(f"Set: {set_name}, F1 Score: {f1_score:.4f}")
    return set_name, f1_score


def main(
    experiment_config_path: str,
    exclude_domain: int | None,
    train_domain: int | None,
) -> None:
    # load experiment config to get data name
    _, experiment_path = get_experiment_name(
        experiment_config_path, MAIN_DIR / "configs" / "experiment"
    )
    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)
    data_cfg_pth = MAIN_DIR / "configs" / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    data_name = data_cfg["name"]

    if data_cfg.get("domain_type") == "exclude_one" and exclude_domain is None:
        msg = (
            "Error: When data domain_type is 'exclude_one', "
            "--exclude-domain argument must be provided."
        )
        raise ValueError(msg)

    if data_cfg.get("domain_type") == "single_domain" and train_domain is None:
        msg = (
            "Error: When data domain_type is 'single_domain', "
            "--train-domain argument must be provided."
        )
        raise ValueError(msg)

    # should be None if neither are defined
    domain_identifier = exclude_domain if exclude_domain is not None else train_domain
    output_path = (
        MAIN_DIR / "outputs" / experiment_path.stem
        if domain_identifier is None
        else MAIN_DIR / "outputs" / experiment_path.stem / f"domain_{domain_identifier}"
    )
    predictions_paths = list(output_path.glob("saved_predictions/*.safetensors"))

    f1_scores = {}

    validation_path = output_path / "saved_predictions" / "validation.safetensors"
    if not validation_path.exists():
        msg = f"Validation predictions not found at {validation_path}"
        raise FileNotFoundError(msg)

    optimised_params = get_best_parameters(validation_path, data_name=data_name)

    for prediction_path in predictions_paths:
        set_name, set_f1 = evaluate_set(
            prediction_data_path=prediction_path,
            data_name=data_name,
            **optimised_params,  # type: ignore[arg-type]
        )
        f1_scores[set_name] = set_f1

    analysis_results = {
        "f1_scores": f1_scores,
        "optimised_parameters": optimised_params,
    }

    output_file_path = output_path / "segment_analysis.yaml"
    with open(output_file_path, "w") as file:
        yaml.dump(analysis_results, file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyse model predictions")
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
    parser.add_argument(
        "--train-domain",
        type=int,
        default=None,
        help="Domain to train when domain_type is 'single_domain'",
    )
    args = parser.parse_args()
    main(
        args.experiment_config,
        exclude_domain=args.exclude_domain,
        train_domain=args.train_domain,
    )
