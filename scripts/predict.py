from pathlib import Path

import yaml

from dr_sad.predicting import load_data_eval, load_model_eval, save_predictions_chunked

MAIN_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def main(
    model_path: str,
    experiment_config: str,
    data_config: str,
    exclude_domain: int | None = None,
) -> None:
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

    print("Saving test predictions...")
    save_predictions_chunked(
        model,
        test_loader,
        prediction_dir / "test_predictions.safetensors",
        chunk_size=50,
    )

    if domain_loader is not None:
        print("Saving excluded domain predictions...")
        save_predictions_chunked(
            model,
            domain_loader,
            prediction_dir / "excluded_domain_predictions.safetensors",
            chunk_size=50,
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
