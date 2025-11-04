import argparse
from pathlib import Path

import yaml
from pytorch_lightning.loggers import CSVLogger
from safetensors.torch import save_model

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import domain_split_dataloaders, from_keys_dataloaders
from dr_sad.training import DrSadTrainer, create_model

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def get_experiment_name(exp_name_arg: str) -> tuple[str, Path]:
    """Get experiment name and path from argument."""
    p = Path(exp_name_arg)
    if p.exists():
        experiment_path = p
    elif (CONFIG_DIR / p).exists():
        experiment_path = CONFIG_DIR / p
    else:
        msg = f"Experiment config not found: {exp_name_arg}"
        raise FileNotFoundError(msg)
    # Determine experiment_name string
    try:
        # If experiment_path is inside CONFIG_DIR
        rel = experiment_path.relative_to(CONFIG_DIR)
        # Remove suffix (.yaml/.yml) and return the parent path + stem
        experiment_name = str(rel.with_suffix("")).replace("\\", "/")
    except ValueError:
        # Not inside CONFIG_DIR → just use the filename without suffix
        experiment_name = experiment_path.stem

    return experiment_name, experiment_path


def main(args) -> None:
    experiment_name, experiment_path = get_experiment_name(args.experiment_name)
    with open(experiment_path) as f:
        exp_config = yaml.safe_load(f)

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
        save_dir = MAIN_DIR / "outputs" / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.exclude_domain is not None:
            err_msg = "Cannot exclude domain when domain_type is set to 'all'."
            raise ValueError(err_msg)

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            batch_size=trainer_cfg["batch_size"],
        )
    elif data_cfg["domain_type"] == "exclude_one":
        save_dir = (
            MAIN_DIR / "outputs" / experiment_name / f"domain_{args.exclude_domain}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
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
            random_seed=exp_config["random_seed"],
        )

    else:
        err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
        raise ValueError(err_msg)

    # Set up logging
    logger = CSVLogger(save_dir=save_dir, name=None)
    log_per_batch = trainer_cfg.get("log_per_batch", 10)
    if log_per_batch < 1:
        msg = f"log_per_batch must be greater than 1, got {log_per_batch}"
        raise ValueError(msg)
    if log_per_batch > len(train_loader):
        # Log every batch if log_per_batch exceeds number of batches
        log_steps = 1
    else:
        log_steps = int(len(train_loader) / log_per_batch)

    # Create trainer with early stopping
    trainer = DrSadTrainer.create_trainer(
        default_root_dir=save_dir,
        logger=logger,
        log_every_n_steps=log_steps,
        max_epochs=trainer_cfg["max_epochs"],
        early_stopping_cfg=trainer_cfg["early_stopping"],
    )

    model = create_model(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate and collect results
    print("Evaluating on Validation data:")
    result_validation = trainer.test(model, val_loader)[0]

    print("Evaluating on In-Domain Test data:")
    result_in_domain = trainer.test(model, test_loader)[0]

    if args.exclude_domain is not None:
        print("Evaluating on Out-of-Domain data:")
        result_out_domain = trainer.test(model, domain_loader)[0]
    else:
        result_out_domain = None

    results = {
        "validation": result_validation,
        "in_domain_test": result_in_domain,
        "out_of_domain_test": result_out_domain,
    }
    with (Path(save_dir) / "test_results.yaml").open("w") as f:
        yaml.safe_dump(results, f)

    # Save model
    trainer.save_checkpoint(Path(save_dir) / "final_checkpoint.ckpt")
    save_model(model, Path(save_dir) / "trained_model_weights.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        default="test",
        help="Name of the experiment config YAML file (without .yaml extension)",
    )
    parser.add_argument(
        "--exclude-domain",
        type=int,
        default=None,
        help="Domain to exclude from training/validation/test (for domain adaptation)",
    )
    args = parser.parse_args()
    main(args)
