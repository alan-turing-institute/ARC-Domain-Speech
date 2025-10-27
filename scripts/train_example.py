"""This script trains the PyanNet model on the CallHome dataset.

Usage:
    python scripts/train_callhome.py --dataset callhome
"""

import argparse
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import from_keys_dataloaders
from dr_sad.pyannet.pyannet import PyanNet

main_dir = Path(__file__).resolve().parent.parent


def main(args) -> None:
    data = load_data(args.dataset)
    with open(main_dir / "data" / str(args.dataset) / "datasplit.yaml") as file:
        data_split = yaml.safe_load(file)

    train_loader, val_loader, test_loader = from_keys_dataloaders(
        data,
        train_keys=data_split["train"],
        val_keys=data_split["val"],
        test_keys=data_split["test"],
        batch_size=4,
    )

    model = PyanNet()

    trainer = Trainer(
        max_epochs=20,
        default_root_dir=main_dir / "outputs" / "example",
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.save_checkpoint(Path(trainer.log_dir) / "final_checkpoint.ckpt")
    torch.save(model.state_dict(), Path(trainer.log_dir) / "trained_model_weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="callhome", help="Dataset name")
    args = parser.parse_args()
    main(args)
