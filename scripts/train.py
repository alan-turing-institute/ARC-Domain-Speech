import argparse
from pathlib import Path

import torch
import yaml

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    domain_split_dataloaders,
    train_test_split_dataloaders,
)
from dr_sad.pyannet.pyannet import PyanNet
from dr_sad.training import TrainerSetup


def main(args) -> None:
    # Load configs from provided paths
    with open(args.trainer_config) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(args.dataset_config) as f:
        dataset_cfg = yaml.safe_load(f)
    """Train PyanNet model with configurable early stopping and LR scheduling."""
    data = load_data(dataset_cfg["name"])

    if args.exclude_domain is not None:
        # First, get initial train/val/test split to generate keys
        temp_train, temp_val, temp_test = train_test_split_dataloaders(
            data,
            batch_size=1,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
        )

        # Extract keys from the temporary datasets
        train_keys = [str(idx) for idx in temp_train.dataset.data.index]
        val_keys = [str(idx) for idx in temp_val.dataset.data.index]
        test_keys = [str(idx) for idx in temp_test.dataset.data.index]

        # Use domain_split_dataloaders to exclude the specified domain
        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            domain=dataset_cfg.get("exclude_domain"),
            batch_size=trainer_cfg["batch_size"],
            random_seed=42,
        )

    else:
        # Use standard random splitting
        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=trainer_cfg["batch_size"],
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
        )
        domain_loader = None

    # Create model with optional scheduler
    if trainer_cfg.get("scheduler", {}).get("type"):
        model = TrainerSetup.create_model_with_scheduler(
            PyanNet,
            scheduler_patience=trainer_cfg["scheduler"].get("patience", 3),
            scheduler_factor=trainer_cfg["scheduler"].get("factor", 0.5),
            learning_rate=trainer_cfg.get("learning_rate", 0.01),
        )
    else:
        model = PyanNet(learning_rate=trainer_cfg.get("learning_rate", 0.01))

    # Create trainer with early stopping
    trainer = TrainerSetup.create_trainer(
        max_epochs=trainer_cfg.get("max_epochs", 25),
        early_stopping_patience=trainer_cfg.get("early_stopping", {}).get(
            "patience", 10
        ),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # If we excluded a domain, also test on that domain for domain adaptation analysis
    if domain_loader is not None:
        trainer.test(model, domain_loader)

    # Save model
    trainer.save_checkpoint(Path(trainer.log_dir) / "final_checkpoint.ckpt")
    torch.save(model.state_dict(), Path(trainer.log_dir) / "trained_model_weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer-config",
        type=str,
        default="configs/training/trainer_config.yaml",
        help="Path to trainer config YAML file",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/training/dataset_config.yaml",
        help="Path to dataset config YAML file",
    )
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude from training/validation/test (for domain adaptation)",
    )
    args = parser.parse_args()
    main(args)
