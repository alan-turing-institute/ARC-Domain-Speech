from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.functional import binary_cross_entropy, cross_entropy

from dr_sad.pyannet import PyanNet
from dr_sad.pyannet.linearnet import LinearNet


class GradientReversalFunction(Function):  # type: ignore[misc]
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # forward is identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse gradient and scale by lambda_
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):  # type: ignore[misc]
    """Layer that applies gradient reversal in the backward pass."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialNet(PyanNet):
    """PyanNet with adversarial domain adaptation."""

    def __init__(
        self,
        *args,
        num_domains: int,
        grl_lambda: float,
        domain_loss_weight: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            num_domains (int): Number of domain classes.
            grl_lambda (float): Strength of the gradient reversal.
            domain_loss_weight (float, optional): Weight for the domain loss term.
            *args, **kwargs: Passed through to PyanNet.
        """
        super().__init__(*args, **kwargs)

        self.num_domains = num_domains
        self.domain_loss_weight = domain_loss_weight

        # gradient reversal layer between LSTM and domain linear block
        self.domain_grl = GradientReversalLayer(lambda_=grl_lambda)

        # second linear + classifier for domain prediction
        self.domain_linear = LinearNet(
            input_dim=self.lstm.out_features,
            **self.hparams.linear,
        )
        self.domain_classifier = nn.Linear(
            self.domain_linear.out_features,
            self.num_domains,
        )

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with both speaker and domain predictions.

        Args:
            waveforms: (batch, channel, samples), channel must be 1.

        Returns:
            speaker_scores: (batch, frames, num_speaker_classes)
            domain_logits: (batch, frames, num_domains)
        """
        # shared feature extractor
        features = self.sincnet(waveforms)
        lstm_out, _ = self.lstm(features)

        # --- speaker head (original path) ---
        speaker_hidden = self.linear(lstm_out)
        speaker_logits = self.classifier(speaker_hidden)
        speaker_scores = self.final_activation(speaker_logits)

        # --- domain head (adversarial path) ---
        domain_feat = self.domain_grl(lstm_out)  # Reverse gradients
        domain_hidden = self.domain_linear(domain_feat)
        domain_logits = self.domain_classifier(domain_hidden)

        return speaker_scores, domain_logits

    def loss_output_function(
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

    def loss_domain_function(
        self,
        _speaker_truth: torch.Tensor,
        domains: list[int],
        domain_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Categorical cross-entropy loss for domain classification.

        Args:
            _speaker_truth (torch.Tensor): Annotations for speaker (not used).
            domains (list[int]): List of domain indices for each sample.
            domain_logits (torch.Tensor): Domain logits from the model.

        Returns:
            loss (torch.Tensor): Computed domain classification loss.
        """
        # replicate per-utterance domain labels over frames
        domain_targets = torch.tensor(
            domains, dtype=torch.long, device=domain_logits.device
        )

        # average domain logits over time to get one prediction per utterance
        domain_logits_mean = domain_logits.mean(dim=1)

        return cross_entropy(domain_logits_mean, domain_targets)

    def _compute_losses(
        self,
        speaker_truth: torch.Tensor,
        domains: list[int],
        speaker_outputs: torch.Tensor,
        domain_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal helper: returns (total, speaker, domain) losses."""
        speaker_loss = self.loss_output_function(
            speaker_truth, domains, speaker_outputs
        )
        domain_loss = self.loss_domain_function(speaker_truth, domains, domain_logits)
        total_loss = speaker_loss + self.domain_loss_weight * domain_loss
        return total_loss, speaker_loss, domain_loss

    def loss_function(  # type: ignore[override]
        self,
        speaker_truth: torch.Tensor,
        domains: list[int],
        speaker_outputs: torch.Tensor,
        domain_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Combined speaker + domain adversarial loss."""
        total_loss, _, _ = self._compute_losses(
            speaker_truth, domains, speaker_outputs, domain_logits
        )
        return total_loss

    def predict_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Override LightningModule predict step"""
        # (batch, time, channels) -> (batch, channels, time)
        outputs, _domain_logits = self(batch["waveforms"])
        return outputs.swapaxes(1, 2)

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
        outputs, domain_logits = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)
        speaker_truth = self.prepare_annotation(waveforms, annotations)
        loss = self.loss_function(speaker_truth, _domains, outputs, domain_logits)
        self.log("train_loss", loss)
        return loss

    def domain_accuracy_function(
        self,
        domains: list[int],
        domain_logits: torch.Tensor,
    ) -> float:
        """
        Computes domain classification accuracy.

        Args:
            domains (list[int] or tensor): True domain index for each sample.
            domain_logits (torch.Tensor): Raw domain logits of shape.

        Returns:
            float: Domain accuracy over the batch.
        """
        # Convert domains to tensor if needed
        if not torch.is_tensor(domains):
            domains = torch.tensor(
                domains, dtype=torch.long, device=domain_logits.device
            )

        pooled_logits = domain_logits.mean(dim=1)
        preds = pooled_logits.argmax(dim=-1)
        correct = (preds == domains).float()
        return float(correct.mean().item())

    def evaluate_batch(  # type: ignore[override]
        self, batch: Any, _batch_idx: int
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
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
            speaker_loss (torch.Tensor): Computed speaker loss for the batch.
            domain_loss (torch.Tensor): Computed domain loss for the batch.
            domain_accuracy (float): Computed domain accuracy for the batch.
        """
        waveforms, annotations, _domains = (
            batch["waveforms"],
            batch["annotations"],
            batch["domains"],
        )
        outputs, domain_logits = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)
        speaker_truth = self.prepare_annotation(waveforms, annotations)
        total_loss, speaker_loss, domain_loss = self._compute_losses(
            speaker_truth, _domains, outputs, domain_logits
        )
        accuracy = self.accuracy_function(speaker_truth, _domains, outputs)
        domain_accuracy = self.domain_accuracy_function(_domains, domain_logits)
        return total_loss, accuracy, speaker_loss, domain_loss, domain_accuracy

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, accuracy, speaker_loss, domain_loss, domain_accuracy = (
            self.evaluate_batch(batch, batch_idx)
        )
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_speaker_loss", speaker_loss)
        self.log("test_domain_loss", domain_loss)
        self.log("test_domain_accuracy", domain_accuracy)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, accuracy, speaker_loss, domain_loss, domain_accuracy = (
            self.evaluate_batch(batch, batch_idx)
        )
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        self.log("val_speaker_loss", speaker_loss)
        self.log("val_domain_loss", domain_loss)
        self.log("val_domain_accuracy", domain_accuracy)
