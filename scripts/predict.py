from pathlib import Path

import yaml

from dr_sad.predicting import load_data_eval, load_model_eval, save_predictions_chunked
from dr_sad.pyannet import PyanNet
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"
CHUNK_SIZE = 25
SAVE_NAMES = {
    "ood": "out_of_domain.safetensors",
    "val": "validation.safetensors",
    "test": "test.safetensors",
}


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
    domain: int | None = None,
    split_idx: int = 0,
) -> None:
    """
    Runs prediction using a trained model on a specified dataset, with optional domain
    specification.

    Args:
        model_path (str): Path to the trained model file (safetensors format).
        experiment_config (str): Name of the experiment configuration file located in
        configs/experiment/.
        domain (int | None, optional): Domain to use/exclude based on domain_type from
        data config. Defaults to None.
    """
    experiment_name, experiment_config_path = get_experiment_name(
        experiment_config, EXP_CONFIG_DIR
    )
    with open(experiment_config_path) as f:
        exp_config = yaml.safe_load(f)

    # default save names
    save_names = dict(SAVE_NAMES)

    # load other configs from experiment config
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]

    # handle the case where model is trained with all domains so weights are stored in
    # main folder of experiment
    train_data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    with open(train_data_cfg_pth) as f:
        train_data_cfg = yaml.safe_load(f)
        train_type = train_data_cfg["domain_type"]
        if train_type == "single_domain" and domain is None:
            save_names["test"] = "test_single.safetensors"

    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    model_cfg_pth = Path(CONFIG_DIR) / "model" / exp_config["model_config"]

    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(model_cfg_pth) as f:
        model_cfg = yaml.safe_load(f)

    domain_name = (
        f"domain_{domain}" if (domain is not None and train_type != "all") else ""
    )

    split_file = (
        MAIN_DIR / "data" / data_cfg["name"] / data_cfg["split_names"][split_idx]
    )
    split_name = split_file.stem

    with split_file.open() as file:
        data_split = yaml.safe_load(file)

    experiment_folder = (
        MAIN_DIR / "outputs" / experiment_name / split_name / domain_name
    )

    model_folder = Path(experiment_folder)

    model = load_model_eval(
        model_path=model_folder / "trained_model_weights.safetensors",
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        data_cfg=data_cfg,
    )

    # Determine how to use domain based on training type
    exclude_domain = domain if train_type == "exclude_one" else None
    train_domain = domain if train_type == "single_domain" else None

    validation_loader, test_loader, domain_loader = load_data_eval(
        data_cfg=data_cfg,
        data_split=data_split,
        trainer_cfg=trainer_cfg,
        exp_config=exp_config,
        exclude_domain=exclude_domain,
        train_domain=train_domain,
    )

    # directory for predictions should stay the same
    prediction_dir = Path(experiment_folder) / "saved_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    save_model_metadata(model, prediction_dir.parent)

    print("Saving validation predictions...")
    save_predictions_chunked(
        model,
        validation_loader,
        prediction_dir / save_names["val"],
        chunk_size=CHUNK_SIZE,
    )
    print("Saving test predictions...")
    save_predictions_chunked(
        model,
        test_loader,
        prediction_dir / save_names["test"],
        chunk_size=CHUNK_SIZE,
    )

    if domain_loader is not None:
        print("Saving excluded domain predictions...")
        save_predictions_chunked(
            model,
            domain_loader,
            prediction_dir / save_names["ood"],
            chunk_size=CHUNK_SIZE,
        )

    if train_type == "all" and domain is not None:
        print("Saving 'single' domain outputs...")
        # directory for predictions should stay the same
        domain_prediction_dir = (
            Path(experiment_folder) / f"domain_{domain}" / "saved_predictions"
        )
        domain_prediction_dir.mkdir(parents=True, exist_ok=True)
        data_name = data_cfg["name"]
        print(
            f"Loading data configs for: {data_name} for domain-specific predictions..."
        )
        single_domain_name = f"{data_name}_single.yaml"
        exclude_domain_name = f"{data_name}_domain.yaml"
        with open(CONFIG_DIR / "data" / single_domain_name) as f:
            single_domain_cfg = yaml.safe_load(f)
        with open(CONFIG_DIR / "data" / exclude_domain_name) as f:
            exclude_domain_cfg = yaml.safe_load(f)

        _, test_loader, _ = load_data_eval(
            data_cfg=single_domain_cfg,
            data_split=data_split,
            trainer_cfg=trainer_cfg,
            exp_config=exp_config,
            train_domain=domain,
            exclude_domain=None,
        )
        save_predictions_chunked(
            model,
            test_loader,
            domain_prediction_dir / "test_single.safetensors",
            chunk_size=CHUNK_SIZE,
        )

        print("Saving 'all_except' domain outputs...")
        _, test_loader, _ = load_data_eval(
            data_cfg=exclude_domain_cfg,
            data_split=data_split,
            trainer_cfg=trainer_cfg,
            exp_config=exp_config,
            train_domain=None,
            exclude_domain=domain,
        )
        save_predictions_chunked(
            model,
            test_loader,
            domain_prediction_dir / "test_all_except.safetensors",
            chunk_size=CHUNK_SIZE,
        )


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
        "split_idx",
        type=int,
        help="Index of the data split to use, read from data config file",
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=None,
        help="Domain to use/exclude based on domain_type from experiment config",
    )

    args = parser.parse_args()

    main(
        experiment_config=args.experiment_config,
        domain=args.domain,
        split_idx=args.split_idx,
    )
