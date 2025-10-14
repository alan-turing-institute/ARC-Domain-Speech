import argparse
from pathlib import Path

import torch
from lightning.pytorch import Trainer

from dr_sad.data.dataloaders import DrSadDataset, get_dataloaders
from dr_sad.pyannet.pyannet import PyanNet


def main(args) -> None:
    dataset = DrSadDataset(args.dataset, domains=["eng"])
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=4, val_ratio=0.1, test_ratio=0.1, random_state=42
    )

    model = PyanNet()

    trainer = Trainer(
        max_epochs=25,
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
