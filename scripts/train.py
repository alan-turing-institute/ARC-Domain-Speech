import argparse
from pathlib import Path

import yaml
from pytorch_lightning.loggers import CSVLogger
from safetensors.torch import save_model

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    domain_split_dataloaders,
    from_keys_dataloaders,
    one_test_dataloader,
    single_domain_dataloaders,
)
from dr_sad.training import DrSadTrainer, create_model
from dr_sad.utils import get_experiment_name

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"
EXP_CONFIG_DIR = CONFIG_DIR / "experiment"


def main(args) -> None:
    """
    Train PyanNet model with configurable early stopping and LR scheduling.

    Args:
        args: An object (typically argparse.Namespace) with the following attributes:
            experiment_name (str): The name or path of the experiment configuration
            file.
            domain (int or None): The domain to use/exclude based on domain_type from
            data config.
    """

    experiment_name, experiment_path = get_experiment_name(
        args.experiment_name, EXP_CONFIG_DIR
    )
    print(f"Running experiment: {experiment_name}")

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

    time_slice = exp_config.get("time_slice")
    if time_slice is not None:
        batch_multiplier = exp_config.get("batch_multiplier")
        if batch_multiplier is None:
            msg = "If time_slice is set, batch_multiplier must also be set."
            raise ValueError(msg)

        batch_size = int(trainer_cfg["batch_size"] * batch_multiplier)
        full_batch_size = trainer_cfg["batch_size"]
    else:
        batch_size = trainer_cfg["batch_size"]
        full_batch_size = None

    data_split_file = (
        MAIN_DIR / "data" / data_cfg["name"] / data_cfg["split_names"][args.split_idx]
    )
    with data_split_file.open() as file:
        data_split = yaml.safe_load(file)

    split_name = data_split_file.stem

    data = load_data(data_cfg["name"], num_workers=trainer_cfg["num_workers"])

    if data_cfg["domain_type"] == "all":
        save_dir = MAIN_DIR / "outputs" / experiment_name / split_name
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.domain is not None:
            err_msg = "Cannot specify domain when domain_type is set to 'all'."
            raise ValueError(err_msg)

        train_loader, val_loader, test_loader = from_keys_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            batch_size=batch_size,
            time_slice=time_slice,
            random_seed=exp_config["random_seed"],
        )
    elif data_cfg["domain_type"] == "single_domain":
        if args.domain is None:
            err_msg = "Must specify --domain when domain_type is 'single_domain'."
            raise ValueError(err_msg)

        save_dir = (
            MAIN_DIR
            / "outputs"
            / experiment_name
            / split_name
            / f"domain_{args.domain}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, test_loader = single_domain_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=args.domain,
            batch_size=batch_size,
            time_slice=time_slice,
            random_seed=exp_config["random_seed"],
        )

    elif data_cfg["domain_type"] == "exclude_one":
        if args.domain is None:
            err_msg = "Must specify --domain when domain_type is 'exclude_one'."
            raise ValueError(err_msg)
        save_dir = (
            MAIN_DIR
            / "outputs"
            / experiment_name
            / split_name
            / f"domain_{args.domain}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        # Use domain_split_dataloaders to exclude the specified domain
        train_loader, val_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=args.domain,
            batch_size=batch_size,
            time_slice=time_slice,
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
        data_cfg=data_cfg,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate and collect results
    print("Evaluating on Validation data:")
    result_validation = trainer.test(model, val_loader)[0]

    print("Evaluating on In-Domain Test data:")
    result_in_domain = trainer.test(model, test_loader)[0]

    if data_cfg["domain_type"] == "exclude_one":
        print("Evaluating on Out-of-Domain data:")
        result_out_domain = trainer.test(model, domain_loader)[0]
    else:
        result_out_domain = None

    results = {
        "validation": result_validation,
        "in_domain_test": result_in_domain,
        "out_of_domain_test": result_out_domain,
    }

    if full_batch_size is not None:
        print("Evaluating with full batch size audio clips")
        if data_cfg["domain_type"] == "all":
            full_test_loader = one_test_dataloader(
                data,
                data_keys=data_split["test"],
                batch_size=full_batch_size,
            )
            results["in_domain_test_full"] = trainer.test(model, full_test_loader)[0]
            results["out_of_domain_test_full"] = None

        elif data_cfg["domain_type"] == "single_domain":
            target_keys = data[data["domains"] == args.domain].index.to_list()
            test_only_target_keys = list(set(data_split["test"]) & set(target_keys))
            full_test_loader = one_test_dataloader(
                data,
                data_keys=test_only_target_keys,
                batch_size=full_batch_size,
            )
            results["in_domain_test_full"] = trainer.test(model, full_test_loader)[0]

        elif data_cfg["domain_type"] == "exclude_one":
            domain_keys = data[data["domains"] == args.domain].index.to_list()
            test_without_domain_keys = list(set(data_split["test"]) - set(domain_keys))
            full_test_loader = one_test_dataloader(
                data,
                data_keys=test_without_domain_keys,
                batch_size=full_batch_size,
            )
            results["in_domain_test_full"] = trainer.test(model, full_test_loader)[0]
            full_domain_loader = one_test_dataloader(
                data,
                data_keys=domain_keys,
                batch_size=full_batch_size,
            )
            results["out_of_domain_test_full"] = trainer.test(
                model, full_domain_loader
            )[0]
        else:
            err_msg = (
                f"Unknown domain_type option: {data_cfg['domain_type']} "
                "This should be unreachable."
            )
            raise ValueError(err_msg)

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
        help="Name of the experiment config YAML file (without .yaml extension)",
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
    main(args)
