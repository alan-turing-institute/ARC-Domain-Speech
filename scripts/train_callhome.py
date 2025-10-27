import argparse
from pathlib import Path

import torch

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    domain_split_dataloaders,
    train_test_split_dataloaders,
)
from dr_sad.pyannet.pyannet import PyanNet
from dr_sad.training import TrainerSetup


def main(args) -> None:
    """Train PyanNet model with configurable early stopping and LR scheduling."""
    data = load_data(args.dataset)

    if args.exclude_domain is not None:
        # First, get initial train/val/test split to generate keys
        temp_train, temp_val, temp_test = train_test_split_dataloaders(
            data, batch_size=1, val_ratio=0.1, test_ratio=0.1, random_seed=42
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
            domain=args.exclude_domain,
            batch_size=args.batch_size,
            random_seed=42,
        )

    else:
        # Use standard random splitting
        train_loader, val_loader, test_loader = train_test_split_dataloaders(
            data,
            batch_size=args.batch_size,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
        )
        domain_loader = None

    # Create model with optional scheduler
    if args.use_scheduler:
        model = TrainerSetup.create_model_with_scheduler(
            PyanNet,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            learning_rate=args.learning_rate,
        )
    else:
        model = PyanNet(learning_rate=args.learning_rate)

    # Create trainer with early stopping
    trainer = TrainerSetup.create_trainer(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
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
    parser.add_argument("--dataset", default="callhome", help="Dataset name")
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude from training/validation/test (for domain adaptation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=25, help="Maximum training epochs"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--use-scheduler", action="store_true", help="Use ReduceLROnPlateau scheduler"
    )
    parser.add_argument(
        "--scheduler-patience", type=int, default=5, help="Scheduler patience"
    )
    parser.add_argument(
        "--scheduler-factor", type=float, default=0.5, help="Scheduler factor"
    )
    args = parser.parse_args()
    main(args)
