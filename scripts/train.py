import argparse
from pathlib import Path

import torch
import yaml

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import domain_split_dataloaders, from_keys_dataloaders
from dr_sad.training import DrSadTrainer, create_model

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def main(args) -> None:
    # Load configs from provided paths
    with open(Path(CONFIG_DIR) / "experiment" / args.experiment_name) as f:
        exp_config = yaml.safe_load(f)
    """Train PyanNet model with configurable early stopping and LR scheduling."""
    trainer_cfg_pth = Path(CONFIG_DIR) / "training" / exp_config["training_config"]
    data_cfg_pth = Path(CONFIG_DIR) / "data" / exp_config["data_config"]
    model_cfg_pth = Path(CONFIG_DIR) / "model" / exp_config["model_config"]

    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)
    with open(model_cfg_pth) as f:
        model_cfg = yaml.safe_load(f)

    data = load_data(data_cfg["name"])
    with open(MAIN_DIR / "data" / data_cfg["name"] / data_cfg["split_name"]) as file:
        data_split = yaml.safe_load(file)

    if data_cfg["domain_type"] == "all":
        domain_identifier = "all"
        if args.exclude_domain is not None:
            err_msg = "Cannot exclude domain when domain_type is set to 'all'."
            raise ValueError(err_msg)

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            batch_size=4,
            # REMOVE THIS
            random_seed=42,
        )
    elif data_cfg["domain_type"] == "exclude_one":
        domain_identifier = f"domain_{args.exclude_domain}"
        if args.exclude_domain is None:
            err_msg = "Must specify --exclude_domain when domain_type is 'exclude_one'."
            raise ValueError(err_msg)
        # Use domain_split_dataloaders to exclude the specified domain
        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=args.exclude_domain,
            batch_size=trainer_cfg["batch_size"],
        )

    else:
        err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
        raise ValueError(err_msg)

    # Create trainer with early stopping
    save_dir = (
        MAIN_DIR
        / "outputs"
        / f"{args.experiment_name.rstrip('.yaml')}"
        / domain_identifier
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = DrSadTrainer.create_trainer(
        default_root_dir=save_dir,
        max_epochs=trainer_cfg["max_epochs"],
        early_stopping_cfg=trainer_cfg["early_stopping"],
    )

    model = create_model(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    if args.exclude_domain is not None:
        print("Evaluating on excluded domain data...")
        trainer.test(model, domain_loader)

    # Save model
    trainer.save_checkpoint(Path(save_dir) / "final_checkpoint.ckpt")
    torch.save(model.state_dict(), Path(save_dir) / "trained_model_weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        default="test",
        help="Name of the experiment config YAML file (without .yaml extension)",
    )
    parser.add_argument(
        "--exclude_domain",
        type=int,
        default=None,
        help="Domain to exclude from training/validation/test (for domain adaptation)",
    )
    args = parser.parse_args()
    main(args)
