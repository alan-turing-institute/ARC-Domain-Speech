from pathlib import Path

import yaml

from dr_sad.predicting import load_data_eval, load_model_eval, save_predictions_chunked
from dr_sad.pyannet import PyanNet
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def save_model_metadata(model: PyanNet, prediction_dir: Path) -> None:
    """
    Save model metadata to a YAML file.

    Args:
        model: The trained model object containing metadata.
        prediction_dir (Path): The path where the metadata YAML file will be saved.
    """
    metadata = {
        "sample_rate": model.sincnet.sample_rate,
        "receptive_field_samples": model.sincnet.receptive_field_size(1),
        "receptive_field_sec": model.sincnet.receptive_field_size(1)
        / model.sincnet.sample_rate,
        "frame_hop_samples": model.sincnet.frame_hop_samples,
        "frame_hop_sec": model.sincnet.frame_hop_samples / model.sincnet.sample_rate,
        "frame_rate_hz": model.sincnet.frame_rate_hz,
        # first frame in input sequence predicted by model
        "frame_center_start": model.sincnet.frame_center_start_step[0],
        # step between frames in input sequence predicted by model
        "frame_center_step": model.sincnet.frame_center_start_step[1],
    }

    metadata_path = prediction_dir / "model_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)


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
    experiment_name, experiment_config_path = get_experiment_name(
        experiment_config, EXP_CONFIG_DIR
    )
    with open(experiment_config_path) as f:
        exp_config = yaml.safe_load(f)

    # load other configs from experiment config
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]

    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    model_cfg_pth = Path(CONFIG_DIR) / "model" / exp_config["model_config"]

    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(model_cfg_pth) as f:
        model_cfg = yaml.safe_load(f)

    domain_name = f"domain_{exclude_domain}" if exclude_domain is not None else ""

    model_path = (
        MAIN_DIR
        / "outputs"
        / experiment_name
        / domain_name
        / "trained_model_weights.safetensors"
    )

    model = load_model_eval(
        model_path=model_path,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
    )

    validation_loader, test_loader, domain_loader = load_data_eval(
        data_cfg=data_cfg,
        data_split=None,
        trainer_cfg=trainer_cfg,
        exp_config=exp_config,
        exclude_domain=exclude_domain,
    )

    prediction_dir = Path(model_path).parent / "saved_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    save_model_metadata(model, prediction_dir.parent)

    print("Saving validation predictions...")
    save_predictions_chunked(
        model,
        validation_loader,
        prediction_dir / "validation.safetensors",
        chunk_size=25,
    )
    print("Saving test predictions...")
    save_predictions_chunked(
        model,
        test_loader,
        prediction_dir / "test.safetensors",
        chunk_size=25,
    )

    if domain_loader is not None:
        print("Saving excluded domain predictions...")
        save_predictions_chunked(
            model,
            domain_loader,
            prediction_dir / "out_of_domain.safetensors",
            chunk_size=25,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict using a trained model on specified dataset."
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        required=True,
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
