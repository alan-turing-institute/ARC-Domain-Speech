from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor


class DrSadTrainer(Trainer):  # type: ignore[misc]
    """
    PyTorch Lightning Trainer factory implementing early stopping and LR scheduling.

    This class provides class methods to create PyTorch Lightning trainers with
    sensible defaults for early stopping and learning rate scheduling.
    """

    @classmethod
    def create_trainer(
        cls,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        **trainer_kwargs: Any,
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
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
                min_delta=0.0,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        return cls(
            max_epochs=max_epochs,
            callbacks=callbacks,
            **trainer_kwargs,
        )

    @classmethod
    def create_model_with_scheduler(
        cls,
        model_class: LightningModule,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.1,
        learning_rate: float = 1e-3,
        **model_kwargs: Any,
    ) -> LightningModule:
        """
        Create a model instance with scheduler configuration.

        Args:
            model_class: The model class to instantiate (e.g., PyanNet)
            scheduler_patience: Number of epochs with no improvement after which
                learning rate will be reduced. Default: 5
            scheduler_factor: Factor by which the learning rate will be reduced.
                Default: 0.1
            learning_rate: Initial learning rate. Default: 1e-3
            **model_kwargs: Additional keyword arguments to pass to the model.

        Returns:
            Model instance with scheduler configuration.
        """
        scheduler_config = {
            "patience": scheduler_patience,
            "factor": scheduler_factor,
            "mode": "min",
            "monitor": "val_loss",
        }

        return model_class(
            scheduler_config=scheduler_config,
            learning_rate=learning_rate,
            **model_kwargs,
        )
