from pathlib import Path

import yaml
from safetensors.torch import load_file, save_file

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import domain_split_dataloaders, from_keys_dataloaders
from dr_sad.pyannet.sincnet import map_sincnet_weights
from dr_sad.training import create_model, save_predictions

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def load_model_eval(model_path, model_cfg, trainer_cfg):
    weightless_model = create_model(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
    )

    # Load the model state dict from the safetensors file
    state_dict = load_file(model_path)
    try:
        # Try to load the state dict normally first
        weightless_model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Check if the error is specifically about missing sincnet.features keys
        error_msg = str(e)
        if (
            "Missing key(s) in state_dict" in error_msg
            and "sincnet.features." in error_msg
        ):
            print("Detected old model format, mapping weights to new architecture...")
            # Apply the mapping for old models
            mapped_state_dict = map_sincnet_weights(state_dict)
            weightless_model.load_state_dict(mapped_state_dict)
            print("Successfully loaded model with weight mapping.")

    # Set model to evaluation mode
    return weightless_model.eval()


def load_data_eval(data_cfg, data_split, trainer_cfg, exp_config, exclude_domain):
    # load data
    data = load_data(data_cfg["name"])
    with open(MAIN_DIR / "data" / data_cfg["name"] / data_cfg["split_name"]) as file:
        data_split = yaml.safe_load(file)

    if data_cfg["domain_type"] == "all":
        if exclude_domain is not None:
            err_msg = "Cannot exclude domain when domain_type is set to 'all'."
            raise ValueError(err_msg)

        _, _, test_loader = from_keys_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            batch_size=trainer_cfg["batch_size"],
            random_seed=exp_config["random_seed"],
        )
        return test_loader, None

    if data_cfg["domain_type"] == "exclude_one":
        if exclude_domain is None:
            err_msg = "Must specify --exclude_domain when domain_type is 'exclude_one'."
            raise ValueError(err_msg)
        # Use domain_split_dataloaders to exclude the specified domain
        _, _, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=exclude_domain,
            batch_size=trainer_cfg["batch_size"],
            random_seed=exp_config["random_seed"],
        )

        return test_loader, domain_loader

    err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
    raise ValueError(err_msg)


def main(model_path, experiment_config, data_config, exclude_domain=None):
    # Load experiment config
    exp_config_path = Path(CONFIG_DIR) / "experiment" / experiment_config
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)

    # load other configs from experiment config
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]
    data_cfg_pth = Path(CONFIG_DIR) / "data" / data_config
    model_cfg_pth = Path(CONFIG_DIR) / "model" / exp_config["model_config"]

    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(model_cfg_pth) as f:
        model_cfg = yaml.safe_load(f)

    model = load_model_eval(
        model_path=model_path,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
    )

    test_loader, domain_loader = load_data_eval(
        data_cfg=data_cfg,
        data_split=None,
        trainer_cfg=trainer_cfg,
        exp_config=exp_config,
        exclude_domain=exclude_domain,
    )

    prediction_dir = Path(model_path).parent / "saved_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(model, test_loader, prediction_dir / "test_predictions.csv")

    if domain_loader is not None:
        save_predictions(
            model,
            domain_loader,
            prediction_dir / "excluded_domain_predictions.csv",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict using a trained model on specified dataset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (safetensors format)",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        required=True,
        help="Experiment configuration file name located in configs/experiments/",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Data configuration file name located in configs/data/",
    )
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude when domain_type is 'exclude_one'",
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        experiment_config=args.experiment_config,
        data_config=args.data_config,
        exclude_domain=args.exclude_domain,
    )
