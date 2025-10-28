from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from dr_sad.pyannet import PyanNet


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
            max_epochs: Maximum number of training epochs. Default: 100
            early_stopping_patience: Number of epochs with no improvement after
                which training will be stopped. Default: 10
            **trainer_kwargs: Additional keyword arguments to pass to the Trainer.

        Returns:
            Configured TrainerSetup instance with early stopping and LR monitoring.
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


def create_model(
    model_cfg: dict[str, Any],
    trainer_cfg: dict[str, Any],
    **model_kwargs: Any,
) -> LightningModule:
    """
    Create a model instance with scheduler configuration.

    Args:
        model_cfg: Model configuration dictionary.
        trainer_cfg: Trainer configuration dictionary.
        learning_rate: Initial learning rate. Default: 1e-3
        **model_kwargs: Additional keyword arguments to pass to the model.

    Returns:
        Model instance with scheduler configuration.
    """

    if model_cfg.get("model_name") == "default_pyannet":
        ModelClass = PyanNet

    else:
        err_msg = f"Unknown model name: {model_cfg.get('model_name')}"
        raise ValueError(err_msg)

    # Create model with optional scheduler
    if trainer_cfg["scheduler"]["enabled"]:
        scheduler_config = trainer_cfg["scheduler"].copy()
        scheduler_config.pop("enabled")
        return ModelClass(
            scheduler_config=scheduler_config,
            learning_rate=trainer_cfg["learning_rate"],
            **model_kwargs,
        )

    # If there is no scheduler
    return ModelClass(learning_rate=trainer_cfg["learning_rate"], **model_kwargs)
