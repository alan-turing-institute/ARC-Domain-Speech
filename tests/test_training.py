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
        monkeypatch.setattr(training, "PyanNet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg = {
            "scheduler": {"type": "reduce_on_plateau", "patience": 3, "factor": 0.5},
            "learning_rate": 1e-4,
        }
        model = training.create_model(model_cfg, trainer_cfg, foo="bar")
        assert isinstance(model, DummyModel)
        assert model.scheduler_config["patience"] == 3
        assert model.learning_rate == 1e-4
        assert model.kwargs["foo"] == "bar"

    def test_create_model_without_scheduler(self, monkeypatch):
        monkeypatch.setattr(training, "PyanNet", DummyModel)
        model_cfg = {"model_name": "default_pyannet"}
        trainer_cfg = {}
        model = training.create_model(model_cfg, trainer_cfg)
        assert isinstance(model, DummyModel)
        assert model.scheduler_config is None
        assert model.learning_rate is None
