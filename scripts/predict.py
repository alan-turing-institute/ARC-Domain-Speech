from pathlib import Path

import yaml
from safetensors.torch import load_model

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import domain_split_dataloaders, from_keys_dataloaders
from dr_sad.training import DrSadTrainer, save_predictions

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def main(model_path, experiment_config, data_config, exclude_domain=None):
    # Load experiment config
    exp_config_path = Path(CONFIG_DIR) / "experiments" / experiment_config
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)

    # load other configs from experiment config
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]

    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)

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
    elif data_cfg["domain_type"] == "exclude_one":
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

    else:
        err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
        raise ValueError(err_msg)

    # Create trainer with early stopping
    trainer = DrSadTrainer.create_trainer(
        default_root_dir=model_directory,
        max_epochs=trainer_cfg["max_epochs"],
        early_stopping_cfg=trainer_cfg["early_stopping"],
    )

    model = load_model(model_path)
