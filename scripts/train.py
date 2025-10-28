import argparse
from pathlib import Path

import torch
import yaml

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import domain_split_dataloaders, from_keys_dataloaders
from dr_sad.pyannet.pyannet import PyanNet
from dr_sad.training import DrSadTrainer

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def main(args) -> None:
    # Load configs from provided paths
    with open(Path(CONFIG_DIR) / "experiment" / f"{args.experiment_name}.yaml") as f:
        exp_config = yaml.safe_load(f)
    """Train PyanNet model with configurable early stopping and LR scheduling."""
    trainer_cfg_pth = (
        Path(CONFIG_DIR) / "training" / f"{exp_config['training_config']}.yaml"
    )
    data_cfg_pth = Path(CONFIG_DIR) / "data" / f"{exp_config['data_config']}.yaml"

    with open(trainer_cfg_pth) as f:
        trainer_cfg = yaml.safe_load(f)
    with open(data_cfg_pth) as f:
        data_cfg = yaml.safe_load(f)

    data = load_data(data_cfg["name"])
    with open(MAIN_DIR / "data" / data_cfg["name"] / "datasplit.yaml") as file:
        data_split = yaml.safe_load(file)

    if args.exclude_domain is None:
        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            batch_size=4,
        )
    else:
        # Use domain_split_dataloaders to exclude the specified domain
        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=args.exclude_domain,
            batch_size=trainer_cfg["batch_size"],
            random_seed=42,
        )

    # Create model with optional scheduler
    if trainer_cfg.get("scheduler", {}).get("type"):
        model = DrSadTrainer.create_model_with_scheduler(
            PyanNet,
            scheduler_patience=trainer_cfg["scheduler"].get("patience", 3),
            scheduler_factor=trainer_cfg["scheduler"].get("factor", 0.5),
            learning_rate=trainer_cfg.get("learning_rate", 0.01),
        )
    else:
        model = PyanNet(learning_rate=trainer_cfg.get("learning_rate", 0.01))

    # Create trainer with early stopping
    trainer = DrSadTrainer.create_trainer(
        default_root_dir=MAIN_DIR / "outputs" / args.experiment_name,
        max_epochs=trainer_cfg.get("max_epochs", 25),
        early_stopping_patience=trainer_cfg.get("early_stopping", {}).get(
            "patience", 10
        ),
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    if args.exclude_domain is not None:
        print("Evaluating on excluded domain data...")
        trainer.test(model, domain_loader)

    # Save model
    trainer.save_checkpoint(Path(trainer.log_dir) / "final_checkpoint.ckpt")
    torch.save(model.state_dict(), Path(trainer.log_dir) / "trained_model_weights.pth")


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
