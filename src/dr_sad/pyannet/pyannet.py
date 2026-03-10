# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Adapted from
# Hervé Bredin - http://herve.niderb.fr
# Updated by ARC

__all__ = ("PyanNet",)

from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

from dr_sad.annotation import speaking_map
from dr_sad.pyannet.linearnet import LinearNet
from dr_sad.pyannet.lstmnet import LSTMNet
from dr_sad.pyannet.sincnet import SincNet


class PyanNet(pl.LightningModule):  # type: ignore[misc]
    """PyanNet Segmentation Model

    SincNet > LSTM > Feed forward > Classifier

    Args:
        sincnet (dict, optional): Keyword arguments passed to the SincNet block.
            Defaults to {"stride": 1}.
        lstm (dict, optional): Keyword arguments passed to the LSTM layer.
            Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
            i.e. two bidirectional layers with 128 units each.
        linear (dict, optional): Keyword arguments used to initialize linear layers.
            Defaults to {"hidden_size": 128, "num_layers": 2},
            i.e. two linear layers with 128 units each.
        sample_rate (int, optional): Audio sample rate. Defaults to 16kHz (16000).
        num_classes (int, optional): Number of output classes. Defaults to 1.
            In this implementation, num_classes is always 1.
    """

    def __init__(
        self,
        sincnet: dict[str, Any] | None = None,
        lstm: dict[str, Any] | None = None,
        linear: dict[str, Any] | None = None,
        sample_rate: int = 16000,
        # task: Task | None = None,
        num_classes: int = 1,
        scheduler_config: dict[str, Any] | None = None,
        learning_rate: float = 1e-3,
        dataloader_length: int | None = None,
    ):
        super().__init__()

        SINCNET_DEFAULTS: dict[str, Any] = {"stride": 10}
        LSTM_DEFAULTS: dict[str, Any] = {
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.0,
        }
        LINEAR_DEFAULTS: dict[str, Any] = {"hidden_size": 128, "num_layers": 2}

        sincnet = {**SINCNET_DEFAULTS, **(sincnet or {})}
        sincnet["sample_rate"] = sample_rate
        lstm = {**LSTM_DEFAULTS, **(lstm or {})}
        linear = {**LINEAR_DEFAULTS, **(linear or {})}

        # Store optimizer and scheduler configuration
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.dataloader_length = dataloader_length

        self.save_hyperparameters(
            "sincnet",
            "lstm",
            "linear",
            "scheduler_config",
            "learning_rate",
            "dataloader_length",
        )

        self.sincnet = SincNet(**self.hparams.sincnet)

        self.lstm = LSTMNet(
            input_size=self.sincnet.out_features,
            **self.hparams.lstm,
        )

        self.linear = LinearNet(
            input_dim=self.lstm.out_features,
            **self.hparams.linear,
        )

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.linear.out_features, self.num_classes)

        self.final_activation = nn.Sigmoid()

    @property
    def sample_rate(self) -> int:
        return int(self.hparams.sincnet["sample_rate"])

    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Args:
            num_samples (int): Number of input samples

        Returns:
            num_frames (int): Number of output frames
        """
        return self.sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Args:
            num_frames (int, optional): Number of frames in the output signal.
                Defaults to 1.

        Returns:
            receptive_field_size (int): Receptive field size.
        """
        return self.sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Args:
            frame (int, optional): Frame index. Defaults to 0.

        Returns:
            receptive_field_center (int): Index of receptive field center.
        """
        return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward
        SincNet > LSTM > Feed forward > Classifier

        Args:
            waveforms (torch.Tensor): This is in the form (batch, channel, sample)
                The channel dimension is currently unused and must be 1.

        Returns:
            scores (torch.Tensor): This returns the scores (batch, frame, classes)
                in this implementation classes is always 1.
        """
        outputs = self.sincnet(waveforms)
        outputs, _ = self.lstm(outputs)
        outputs = self.linear(outputs)
        outputs = self.classifier(outputs)
        return self.final_activation(outputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and optionally a learning rate scheduler.

        Returns:
            Optimizer or dictionary with optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # If scheduler config is provided, add ReduceLROnPlateau scheduler
        if self.scheduler_config is not None:
            # Separate PyTorch scheduler params from Lightning config params
            scheduler_params = {
                k: v for k, v in self.scheduler_config.items() if k != "monitor"
            }
            monitor_metric = self.scheduler_config.get("monitor", "val_loss")

            scheduler_type = scheduler_params.pop("type")
            if scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_params
                )
                interval = "epoch"
                frequency = 1
            elif scheduler_type == "CyclicLR":
                scheduler_params["base_lr"] = self.learning_rate

                # Handle step size configuration
                step_epochs = scheduler_params.pop("step_size_up_epoch", None)

                if step_epochs is not None:
                    # Using epoch-based configuration
                    if self.dataloader_length is None:
                        err_msg = (
                            "step_size_up_epoch requires dataloader_length to be set."
                        )
                        raise ValueError(err_msg)
                    scheduler_params["step_size_up"] = (
                        step_epochs * self.dataloader_length
                    )
                elif "step_size_up" not in scheduler_params:
                    # Neither step_size_up_epoch nor step_size_up provided, use default

                    err_msg = (
                        "CyclicLR requires either step_size_up_epoch or "
                        "step_size_up to be set in scheduler_config.\n"
                    )
                    raise ValueError(err_msg)

                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer, **scheduler_params
                )
                interval = "step"
                frequency = 1

            else:
                err_msg = f"Unknown scheduler type: {scheduler_type}"
                raise ValueError(err_msg)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": interval,
                    "frequency": frequency,
                },
            }

        return optimizer

    def frame_centers_start_step(self) -> tuple[float, float]:
        """Compute the start and step of the receptive field samples.
        Assumes symmetric padding (single int per layer) and ceil_mode=False for pools.

        Returns:
            start (float): Input-sample index of the center of output frame 0
            step (float): Effective stride from input samples to output frames
        """
        start = self.sincnet.frame_center_start_step[0] / self.sample_rate
        step = self.sincnet.frame_center_start_step[1] / self.sample_rate
        return start, step

    def frame_centers(
        self, num_samples: int, as_numpy: bool = False
    ) -> list[float] | np.ndarray:
        """
        Vectorized centers for all output frames (in input-sample indices) in seconds
        as floats.

        Args:
            num_samples (int): Number of samples in the input signal

        Returns:
            list[float]: List of frame centers in seconds.
        """
        centers: np.ndarray = self.sincnet.frame_centers(num_samples, as_numpy=True)
        if as_numpy:
            return centers / self.sample_rate
        return (centers / self.sample_rate).tolist()

    def speaking_map(
        self, input_size: int, annotations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Generate speaking map from annotations.

        Args:
            input_size (int): Number of input samples.
            annotations (list[tuple[float, float]]): List of (start, end) times in
                seconds for each period of speech. This list must be ordered and
                non-overlapping.

        """
        frame_centers = self.frame_centers(input_size, as_numpy=True)
        return speaking_map(frame_centers, annotations)

    def prepare_annotation(
        self, waveforms: torch.Tensor, annotations: list[list[tuple[float, float]]]
    ) -> torch.Tensor:
        """Prepare annotation for training/validation step

        Args:
            waveforms (torch.Tensor): Input waveforms. This is used to get the
                number of samples and the device.
            annotations (list[list[tuple[float, float]]]): List of annotations for
                each sample.

        Returns:
            speaker_truth (torch.Tensor): Prepared annotation in shame shape as the
                model output (batch, 1, frames).
        """
        speaker_truths = np.empty(
            (waveforms.shape[0], self.num_frames(waveforms.shape[-1])), dtype=np.float32
        )
        for i, annotation in enumerate(annotations):
            speaker_truths[i] = self.speaking_map(waveforms.shape[-1], annotation)

        speaker_truth = torch.tensor(speaker_truths, device=waveforms.device).float()
        # Map annotations from (batch, frames) to (batch, 1, frames)
        return speaker_truth.unsqueeze(1)

    def loss_function(
        self,
        speaker_truth: torch.Tensor,
        _domains: list[int],
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss function

        Args:
            speaker_truth (torch.Tensor): Ground truth annotations.
            _domains (list[int]): List of domain indices for each sample.
            outputs (torch.Tensor): Model outputs.

        Returns:
            loss (torch.Tensor): Computed loss.
        """
        return binary_cross_entropy(outputs, speaker_truth)

    def accuracy_function(
        self,
        speaker_truth: torch.Tensor,
        _domains: list[int],
        outputs: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Accuracy function

        Args:
            speaker_truth (torch.Tensor): Ground truth annotations.
            _domains (list[int]): List of domain indices for each sample.
            outputs (torch.Tensor): Model outputs.
            threshold (float, optional): Threshold to apply to the outputs.
                Defaults to 0.0.

        Returns:
            accuracy (float): Computed accuracy.
        """
        predictions = (outputs > threshold).float()
        correct = torch.isclose(predictions, speaker_truth).float().sum()
        total = torch.numel(speaker_truth)
        return float((correct / total).item())

    def predict_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Override LightningModule predict step"""
        # (batch, time, channels) -> (batch, channels, time)
        return self(batch["waveforms"]).swapaxes(1, 2)

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Override LightningModule training step

        Args:
            batch (Any): Batch from the dataloader. Includes:
              - waveforms (torch.Tensor): Input waveforms (batch, channel, samples)
                    padded to the same length at the end with zeros.
              - annotations (list[list[tuple[float, float]]]): List of annotations
                    for each sample in the batch.
              - _domains (list[int]): List of domain indices for each sample.
            _batch_idx (int): Batch index, unused.

        Returns:
            loss (torch.Tensor): Computed loss for the batch.
        """
        waveforms, annotations, _domains = (
            batch["waveforms"],
            batch["annotations"],
            batch["domains"],
        )
        outputs = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)
        speaker_truth = self.prepare_annotation(waveforms, annotations)
        loss = self.loss_function(speaker_truth, _domains, outputs)
        self.log("train_loss", loss)
        return loss

    def evaluate_batch(self, batch: Any, _batch_idx: int) -> tuple[torch.Tensor, float]:
        """Function for overriding LightningModule validation_step and test_step

        Args:
            batch (Any): Batch from the dataloader. Includes:
              - waveforms (torch.Tensor): Input waveforms (batch, channel, samples)
                    padded to the same length at the end with zeros.
              - annotations (list[list[tuple[float, float]]]): List of annotations
                    for each sample in the batch.
              - _domains (list[int]): List of domain indices for each sample.

        Logs:
            loss (torch.Tensor): Computed loss for the batch.
            accuracy (float): Computed accuracy for the batch.
        """
        waveforms, annotations, _domains = (
            batch["waveforms"],
            batch["annotations"],
            batch["domains"],
        )
        outputs = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)
        speaker_truth = self.prepare_annotation(waveforms, annotations)
        loss = self.loss_function(speaker_truth, _domains, outputs)
        accuracy = self.accuracy_function(speaker_truth, _domains, outputs)
        return loss, accuracy

    def test_step(self, batch: Any, _batch_idx: int) -> None:  # noqa: PT019
        loss, accuracy = self.evaluate_batch(batch, _batch_idx)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def validation_step(self, batch: Any, _batch_idx: int) -> None:
        loss, accuracy = self.evaluate_batch(batch, _batch_idx)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
