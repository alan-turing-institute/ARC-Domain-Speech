from typing import Any

import pytest
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from dr_sad import training


class DummyModel:
    def __init__(self, scheduler_config=None, learning_rate=None, **kwargs):
        self.scheduler_config = scheduler_config
        self.learning_rate = learning_rate
        self.kwargs = kwargs


class TestDrSadTrainer:
    def test_create_trainer_with_early_stopping(self):
        es_cfg = {"enabled": True, "monitor": "val_loss", "patience": 2}
        trainer = training.DrSadTrainer.create_trainer(
            max_epochs=1,
            early_stopping_cfg=es_cfg,
            logger=False,
            enable_progress_bar=False,
        )
        cb_types = {type(c) for c in trainer.callbacks}
        assert EarlyStopping in cb_types
        assert LearningRateMonitor in cb_types

    def test_create_trainer_without_early_stopping(self):
        trainer = training.DrSadTrainer.create_trainer(
            max_epochs=1,
            early_stopping_cfg={"enabled": False},
            logger=False,
            enable_progress_bar=False,
        )
        cb_types = {type(c) for c in trainer.callbacks}
        assert EarlyStopping not in cb_types
        assert LearningRateMonitor in cb_types


class TestCreateModel:
    def test_create_model_with_scheduler(self, monkeypatch):
        monkeypatch.setitem(training.MODEL_DICT, "default_pyannet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg = {
            "scheduler": {
                "enabled": True,
                "type": "ReduceLROnPlateau",
                "patience": 3,
                "factor": 0.5,
            },
            "learning_rate": 1e-4,
        }
        model = training.create_model(model_cfg, trainer_cfg, foo="bar")
        assert isinstance(model, DummyModel)
        assert model.scheduler_config == {
            "type": "ReduceLROnPlateau",
            "patience": 3,
            "factor": 0.5,
        }
        assert model.learning_rate == 1e-4
        assert model.kwargs["foo"] == "bar"

    def test_create_model_without_scheduler(self, monkeypatch):
        monkeypatch.setitem(training.MODEL_DICT, "default_pyannet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg: dict[str, Any] = {
            "scheduler": {"enabled": False},
            "learning_rate": 1e-4,
        }
        model = training.create_model(model_cfg, trainer_cfg, foo="bar")
        assert isinstance(model, DummyModel)
        assert model.scheduler_config is None
        assert model.learning_rate == 1e-4
        assert model.kwargs["foo"] == "bar"

    def test_create_model_with_irm_model(self, monkeypatch):
        monkeypatch.setitem(training.MODEL_DICT, "irm_model", DummyModel)
        model_cfg = {"model_name": "irm_model", "lambda_irm": 200.0}
        trainer_cfg = {
            "scheduler": {"enabled": True, "type": "ReduceLROnPlateau", "patience": 2},
            "learning_rate": 1e-3,
        }
        model = training.create_model(model_cfg, trainer_cfg)
        assert isinstance(model, DummyModel)
        assert model.scheduler_config == {"type": "ReduceLROnPlateau", "patience": 2}
        assert model.learning_rate == 1e-3
        assert model.kwargs["lambda_irm"] == 200.0

    def test_create_model_with_bad_config(self, monkeypatch):
        monkeypatch.setattr(training, "PyanNet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg: dict[str, Any] = {}
        with pytest.raises(ValueError, match="trainer_cfg cannot be empty"):
            training.create_model(model_cfg, trainer_cfg, foo="bar")

    def test_create_model_missing_scheduler_key(self, monkeypatch):
        monkeypatch.setattr(training, "PyanNet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg: dict[str, Any] = {"learning_rate": 1e-4}
        with pytest.raises(KeyError):
            training.create_model(model_cfg, trainer_cfg, foo="bar")

    def test_unknown_model_name(self):
        model_cfg = {"model_name": "unknown_model"}
        trainer_cfg: dict[str, Any] = {
            "scheduler": {"enabled": False},
            "learning_rate": 1e-4,
        }
        with pytest.raises(ValueError, match="Unknown model name: unknown_model"):
            training.create_model(model_cfg, trainer_cfg, foo="bar")
