from typing import Any

import torch
from torch import Tensor

from dr_sad.pyannet import PyanNet
from dr_sad.utils import TrainingBatch


class VRExModel(PyanNet):
    def __init__(
        self,
        lambda_vrex: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_vrex = lambda_vrex
        self.save_hyperparameters()

    def vrex_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        domain_ids: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute the VREx loss.

        Args:
            logits: Model output logits of shape (batch_size, num_classes, num_frames).
            labels: Ground truth labels of shape (batch_size, 1, num_frames).
            domain_ids: Domain/environment IDs of shape (batch_size,).

        Returns:
            A tuple containing:
            - The total loss (ERM loss + VREx penalty).
            - A dictionary with 'erm_loss' and 'vrex_penalty' metrics.
        """
        # Compute base ERM loss across all samples
        erm_loss: Tensor = self.loss_function(labels, domain_ids, logits)

        # Get unique domains in this batch
        unique_domains = torch.unique(domain_ids)

        # Compute per-domain losses
        domain_losses = []
        for domain in unique_domains:
            # Get mask for samples from this domain
            domain_mask = domain_ids == domain

            # Compute loss for this domain
            domain_logits = logits[domain_mask]
            domain_labels = labels[domain_mask]
            domain_loss = self.loss_function(
                domain_labels, domain_ids[domain_mask], domain_logits
            )
            domain_losses.append(domain_loss)

        # Stack domain losses and compute variance
        domain_losses_tensor = torch.stack(domain_losses)
        if len(domain_losses_tensor) > 1:
            penalty = domain_losses_tensor.var()
        else:
            # No variance with a single domain, prevents pytorch warning
            penalty = torch.zeros(
                1, device=domain_losses_tensor.device, dtype=domain_losses_tensor.dtype
            )

        # Total VREx loss
        total_loss = erm_loss + self.lambda_vrex * penalty

        metrics = {
            "erm_loss": erm_loss.detach(),
            "vrex_penalty": penalty.detach(),
        }

        return total_loss, metrics

    def training_step(
        self,
        batch: TrainingBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Override training step to use V-REx loss"""
        waveforms, annotations, domains = (
            batch["waveforms"],
            batch["annotations"],
            batch["domains"],
        )

        # Forward pass
        outputs = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)

        # Prepare ground truth
        speaker_truth = self.prepare_annotation(waveforms, annotations)

        # Compute VREX loss - VREX loss handles the reshaping
        loss, metrics = self.vrex_loss(outputs, speaker_truth, domains)

        # Log metrics
        self.log("train_loss", loss)
        self.log("train_erm", metrics["erm_loss"])
        self.log("train_vrex_penalty", metrics["vrex_penalty"])
        self.log("lambda_vrex", self.lambda_vrex)

        return loss

    def evaluate_batch_vrex(
        self, batch: Any, _batch_idx: int
    ) -> tuple[Tensor, Tensor, Tensor, float]:
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

        total_loss, metrics = self.vrex_loss(outputs, speaker_truth, _domains)

        erm_loss, vrex_penalty = metrics["erm_loss"], metrics["vrex_penalty"]
        accuracy = self.accuracy_function(speaker_truth, _domains, outputs)
        return total_loss, erm_loss, vrex_penalty, accuracy

    def test_step(self, batch: Any, batch_idx: int) -> None:
        total_loss, accuracy = self.evaluate_batch(batch, batch_idx)
        self.log("test_loss", total_loss)
        self.log("test_accuracy", accuracy)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        total_loss, erm_loss, vrex_penalty, accuracy = self.evaluate_batch_vrex(
            batch, batch_idx
        )
        self.log("val_loss", total_loss)
        self.log("val_erm", erm_loss)
        self.log("val_vrex_penalty", vrex_penalty)
        self.log("val_accuracy", accuracy)
