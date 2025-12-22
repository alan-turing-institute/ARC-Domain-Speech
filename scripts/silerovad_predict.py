from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dr_sad.models.SileroVAD_utils import get_probs, load_silerovad_model
from dr_sad.predicting import load_data_eval
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def main(
    experiment_config: str,
    exclude_domain: int | None = None,
) -> None:
    """
    Runs prediction using a trained model on a specified dataset, with optional domain
    exclusion.

    Args:
        model_path (str): Path to the trained model file (safetensors format).
        experiment_config (str): Name of the experiment configuration file located in
        configs/experiment/.
        data_config (str): Name of the data configuration file located in configs/data/.
        exclude_domain (int | None, optional): Domain to exclude when domain_type is
        'exclude_one'. Defaults to None.
    """
    # Load experiment config
    experiment_name, experiment_config_path = get_experiment_name(
        experiment_config, EXP_CONFIG_DIR
    )
    with open(experiment_config_path) as f:
        exp_config = yaml.safe_load(f)

    # load other configs from experiment config
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)

    # set batch size to 1 for SileroVAD
    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
        trainer_cfg["batch_size"] = 1

    # Create prediction directory
    domain_name = f"domain_{exclude_domain}" if exclude_domain is not None else ""
    prediction_dir = (
        MAIN_DIR
        / "outputs"
        / f"{experiment_name}_silerovad"
        / domain_name
        / "saved_predictions"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Load SileroVAD model and defined prediction parameters
    sample_rate = 16000
    model, vad_iterator, model_metadata = load_silerovad_model(sample_rate)
    metadata_path = prediction_dir.parent / "model_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(model_metadata, f)

    # Load data loaders for evaluation
    validation_loader, test_loader, domain_loader = load_data_eval(
        data_cfg=data_cfg,
        data_split=None,
        trainer_cfg=trainer_cfg,
        exp_config=exp_config,
        exclude_domain=exclude_domain,
    )

    # Iterate over each data loader and perform predictions
    for loader, split in zip(
        [validation_loader, test_loader, domain_loader],
        ["validation", "test", "out_of_domain"],
        strict=True,
    ):
        print(f"Processing {split} set...")

        predictions = {}

        # Iterate over data loader and get predictions
        for _, batch in enumerate(tqdm(loader)):
            file_id = batch["file_id"][0]
            waveform = batch["waveforms"][0]
            probs = get_probs(
                model,
                vad_iterator,
                waveform,
                int(model_metadata["frame_hop_samples"]),
                int(model_metadata["sample_rate"]),
            )
            predictions[file_id] = torch.tensor(probs)

        # save predictions
        prediction_path = prediction_dir / f"{split}.safetensors"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(predictions, prediction_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict using a trained model on specified dataset."
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="Experiment configuration file name located in configs/experiments/",
    )
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude when domain_type is 'exclude_one'",
    )

    args = parser.parse_args()

    main(
        experiment_config=args.experiment_config,
        exclude_domain=args.exclude_domain,
    )
