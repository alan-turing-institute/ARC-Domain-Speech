from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from safetensors.torch import load_file
from tqdm import tqdm

from dr_sad.analysis import load_audio_and_annotations
from dr_sad.annotation import speaking_map
from dr_sad.evaluating import SpeechDetectionEvaluator
from dr_sad.plotting import (
    plot_general_precision_recall_curve,
    plot_precision_recall_curve,
)
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def get_ground_truth(
    model_metadata: dict[str, float | int],
    data_dir: Path,
    file_id: str,
    predictions,
) -> tuple[np.ndarray, np.ndarray]:
    audio, _, speech_segments = load_audio_and_annotations(
        file_id,
        data_dir,
    )

    # get model frame parameters
    frame_rate_hz = model_metadata["frame_rate_hz"]
    frame_center_start = model_metadata["frame_center_start"]
    frame_center_step = model_metadata["frame_center_step"]
    actual_num_frames = int(
        ((len(audio) - 2 * frame_center_start) // frame_center_step) + 1
    )

    # get only the valid portion of predictions
    signal_predictions = predictions[:actual_num_frames]

    all_timestamps = (
        np.arange(len(signal_predictions)) * (1 / frame_rate_hz)
    ) + model_metadata["frame_hop_sec"]
    # Create ground truth mask
    ground_truth_mask = speaking_map(
        timestamps=all_timestamps,
        annotations=speech_segments,
    )
    return ground_truth_mask, signal_predictions


def main(
    experiment_config: str,
) -> None:
    """
    Main function to generate precision-recall curve for model predictions.

    Args:
        experiment_config (str): Path or name to the experiment configuration file.
    """

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
    split_names: list[str] = data_cfg["split_names"]

    # Find all domain directories
    base_experiment_output = (
        MAIN_DIR / "outputs" / experiment_name / split_names[0].rstrip(".yaml")
    )
    domain_dirs = list(base_experiment_output.glob("domain_*"))
    domains = [int(d.name.replace("domain_", "")) for d in domain_dirs]

    # Initialize dictionary to store all results across domains
    all_results: dict[str, dict[str, dict[str, list[float]]]] = {}

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

                n_predictions = len(predictions)
                n_thresholds = 50

                precision = np.zeros((n_thresholds, n_predictions))
                recall = np.zeros((n_thresholds, n_predictions))

                for prediction_idx, (file_id, prediction_tensor) in enumerate(
                    predictions.items()
                ):
                    numpy_prediction = prediction_tensor.numpy().flatten()
                    ground_truth, signal_predictions = get_ground_truth(
                        model_metadata=model_metadata,
                        data_dir=MAIN_DIR / "data" / data_name,
                        file_id=file_id,
                        predictions=numpy_prediction,
                    )
                    for threshold_idx, threshold in enumerate(
                        np.linspace(0, 1, n_thresholds)
                    ):
                        evaluator = SpeechDetectionEvaluator(
                            detection_threshold=threshold,
                        )
                        metrics_dict = evaluator.calculate_base_metrics(
                            ground_truth, signal_predictions
                        )
                        precision[threshold_idx, prediction_idx] = metrics_dict[
                            "true_positives"
                        ] / (
                            metrics_dict["true_positives"]
                            + metrics_dict["false_positives"]
                        )
                        recall[threshold_idx, prediction_idx] = metrics_dict[
                            "true_positives"
                        ] / (
                            metrics_dict["true_positives"]
                            + metrics_dict["false_negatives"]
                        )
                # Save precision-recall data
                results_dict[eval_split]["precision"].append(np.mean(precision, axis=1))
                results_dict[eval_split]["recall"].append(np.mean(recall, axis=1))

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

        # save precision-recall curve to matplotlib figure
        fig, ax = plt.subplots()
        for eval_split in eval_split_names:
            ax = plot_precision_recall_curve(
                stacked_results[eval_split], eval_split, ax
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

        fig.savefig(
            figure_save_path / f"precision_recall_curve_domain_{domain}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    # Create general precision-recall curve across all domains
    plot_general_precision_recall_curve(all_results, figure_save_path)

    # Save all results to files
    results_save_path = figure_save_path / "precision_recall_results.yaml"
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
