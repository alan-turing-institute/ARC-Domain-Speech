from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from dr_sad.data.data_fetching import DOMAIN_SETTINGS
from dr_sad.models import AdversarialNet
from dr_sad.models.IRM import IRMModel
from dr_sad.pyannet import PyanNet

# model registry
MODEL_DICT: dict[str, type[LightningModule]] = {
    "default_pyannet": PyanNet,
    "irm_model": IRMModel,
    "adversarial_net": AdversarialNet,
}


def save_predictions(
    model: LightningModule,
    dataloader: DataLoader,
    save_path: str,
) -> None:
    """
    Iterates over the dataloader, runs predictions using the model's predict_step,
    collects outputs indexed by file_id, and saves them using safetensors.
    Args:
        model (LightningModule): The trained model used for prediction.
        dataloader (Any): An iterable of batches containing input data.
        save_path (str): Path to save the predictions file.
    """
    # Save test predictions
    outputs = {}
    for b_i, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Predicting outputs"
    ):
        file_ids = batch["file_id"]
        prediction: torch.Tensor = model.predict_step(batch, b_i)
        for index, file_id in enumerate(file_ids):
            outputs[file_id] = prediction[index].cpu()

    save_file(outputs, save_path)


class DrSadTrainer(Trainer):  # type: ignore[misc]
    """
    PyTorch Lightning Trainer factory implementing early stopping and LR scheduling.

    This class provides class methods to create PyTorch Lightning trainers with
    sensible defaults for early stopping and learning rate scheduling.
    """

    @classmethod
    def create_trainer(
        cls, max_epochs: int, early_stopping_cfg: dict[str, Any], **trainer_kwargs: Any
    ) -> "DrSadTrainer":
        """
        Create a PyTorch Lightning Trainer with early stopping and LR monitoring.

        Args:
            max_epochs: Maximum number of training epochs.
            early_stopping_cfg: Dictionary containing early stopping configuration.
                Must include an 'enabled' key (bool). If enabled, other keys are passed
                    to EarlyStopping, e.g.
                    {'enabled': True, 'patience': 10, 'monitor': 'val_loss', ...}.
            **trainer_kwargs: Additional keyword arguments to pass to the Trainer.

        Returns:
            Configured DrSadTrainer instance with early stopping and LR monitoring.
        """
        if early_stopping_cfg["enabled"]:
            es_cfg = early_stopping_cfg.copy()
            es_cfg.pop("enabled")
            callbacks = [
                EarlyStopping(**es_cfg),
                LearningRateMonitor(logging_interval="epoch"),
            ]

        else:
            callbacks = [LearningRateMonitor(logging_interval="epoch")]

        return cls(
            max_epochs=max_epochs,
            callbacks=callbacks,
            **trainer_kwargs,
        )


def _get_domain_num_from_data_cfg(data_cfg: dict[str, Any] | None) -> int:
    """
    Get the number of domains from the data configuration.

    Args:
        data_cfg: Data configuration dictionary. Must contain a 'name' key.

    Returns:
        Number of domains as an integer.
    """
    if data_cfg is None:
        err_msg = "data_cfg cannot be None"
        raise ValueError(err_msg)

    domain_name = data_cfg.get("name")
    if domain_name is None:
        err_msg = "data_cfg must contain a 'name' key"
        raise KeyError(err_msg)

    if domain_name not in DOMAIN_SETTINGS:
        err_msg = f"Unknown data_cfg name: {domain_name}"
        raise ValueError(err_msg)

    return len(DOMAIN_SETTINGS[domain_name]["domains_idx"])


def create_model(
    model_cfg: dict[str, Any],
    trainer_cfg: dict[str, Any],
    data_cfg: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> LightningModule:
    """
    Create a model instance with scheduler configuration.

    Args:
        model_cfg: Model configuration dictionary.
        trainer_cfg: Trainer configuration dictionary. Must contain a 'learning_rate'
            key (initial learning rate, default: 1e-3).
        data_cfg: Data configuration dictionary (optional).
        **model_kwargs: Additional keyword arguments to pass to the model.

    Returns:
        Model instance with scheduler configuration.
    """

    if trainer_cfg == {} or trainer_cfg is None:
        err_msg = "trainer_cfg cannot be empty"
        raise ValueError(err_msg)
    if "scheduler" not in trainer_cfg:
        err_msg = "trainer_cfg must contain a 'scheduler' key"
        raise KeyError(err_msg)

    model_name = model_cfg.get("model_name")
    if model_name not in MODEL_DICT:
        err_msg = (
            f"Unknown model name: {model_name}. Available: {list(MODEL_DICT.keys())}"
        )
    for key, value in model_cfg.items():
        if key != "model_name":
            model_kwargs[key] = value

    ModelClass = MODEL_DICT[model_name]
    if model_cfg.get("model_name") == "adversarial_net":
        num_domains = _get_domain_num_from_data_cfg(data_cfg)
        model_kwargs["num_domains"] = num_domains

    # Extract all model config except 'model_name'
    model_specific_kwargs = {k: v for k, v in model_cfg.items() if k != "model_name"}

    # Create model with optional scheduler
    if trainer_cfg["scheduler"]["enabled"]:
        scheduler_config = trainer_cfg["scheduler"].copy()
        scheduler_config.pop("enabled")
        return ModelClass(
            scheduler_config=scheduler_config,
            learning_rate=trainer_cfg["learning_rate"],
            **model_specific_kwargs,
            **model_kwargs,
        )

    # If there is no scheduler
    return ModelClass(
        learning_rate=trainer_cfg["learning_rate"],
        **model_specific_kwargs,
        **model_kwargs,
    )
