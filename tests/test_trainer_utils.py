from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from dr_sad.training.trainer_utils import TrainerSetup


class DummyModel:
    def __init__(self, scheduler_config=None, learning_rate=None, **kwargs):
        self.scheduler_config = scheduler_config
        self.learning_rate = learning_rate
        self.extra_kwargs = kwargs


def test_create_trainer_callbacks():
    trainer = TrainerSetup.create_trainer(max_epochs=5, early_stopping_patience=2)
    # Check Trainer type
    assert isinstance(trainer, Trainer)
    # Check callbacks
    callback_types = [type(cb) for cb in trainer.callbacks]
    assert EarlyStopping in callback_types
    assert LearningRateMonitor in callback_types
    # Check EarlyStopping config
    early_stopping = next(
        cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
    )
    assert early_stopping.patience == 2
    assert early_stopping.monitor == "val_loss"


def test_create_model_with_scheduler():
    model = TrainerSetup.create_model_with_scheduler(
        DummyModel,
        scheduler_patience=3,
        scheduler_factor=0.5,
        learning_rate=0.01,
        extra_arg="test",
    )
    # Check scheduler config
    assert model.scheduler_config["patience"] == 3
    assert model.scheduler_config["factor"] == 0.5
    assert model.scheduler_config["monitor"] == "val_loss"
    assert model.learning_rate == 0.01
    assert model.extra_kwargs["extra_arg"] == "test"
