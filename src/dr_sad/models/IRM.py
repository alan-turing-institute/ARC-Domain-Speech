from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_sad.pyannet import PyanNet


class TrainingBatch(TypedDict):
    waveforms: torch.Tensor
    annotations: list[list[tuple[float, float]]]
    domains: torch.Tensor


class IRMLoss(nn.Module):  # type: ignore[misc]
    """IRM Loss implementation for domain-invariant speaker diarization."""

    def __init__(self, lambda_irm: float = 1e2) -> None:
        super().__init__()
        self.lambda_irm = float(lambda_irm)
        self.dummy_w = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            logits: (batch, channels, frames) - as output from PyanNet
            labels: (batch, channels, frames) - as output from prepare_annotation
            env_ids: (batch,) - environment ID for each sample

        Returns:
            loss: scalar
            metrics: dict with 'erm_loss' and 'irm_penalty'
        """
        # Handle PyanNet output format: (batch, channels, frames)
        # Convert to (batch, frames, channels) and squeeze labels
        logits = logits.transpose(1, 2)  # (batch, frames, channels)
        labels = labels.squeeze(1)  # (batch, frames)
        unique_envs = env_ids.unique()

        # Collect per-environment losses to avoid inefficient tensor accumulation
        env_erm_losses = []
        env_penalties = []

        # Compute ERM loss and IRM penalty for each environment to capture
        # per-environment behavior
        for env in unique_envs:
            # Get samples from this environment
            mask = env_ids == env
            env_logits = logits[mask]  # (batch, frames, channels)
            env_labels = labels[mask]  # (batch, frames)

            # Scale by dummy classifier -> creates computational graph
            env_logits_scaled = env_logits * self.dummy_w

            # For binary classification: logits are (batch, frames, channels),
            # labels are (batch, frames)
            # Squeeze the last dimension from logits to match labels
            env_logits_flat = env_logits_scaled.squeeze(-1)  # (batch, frames)
            env_labels_flat = env_labels.float()

            # Compute loss from SCALED logits (this creates the computational graph for IRM)
            scaled_loss = F.binary_cross_entropy_with_logits(
                env_logits_flat, env_labels_flat
            )

            # For logging: compute ERM loss from original logits
            erm_loss = F.binary_cross_entropy_with_logits(
                env_logits.squeeze(-1), env_labels_flat
            )
            env_erm_losses.append(erm_loss)

            # IRM penalty: gradient of scaled_loss w.r.t. dummy_w
            grad = torch.autograd.grad(
                scaled_loss, self.dummy_w, create_graph=True, retain_graph=True
            )[0]
            penalty = grad**2
            env_penalties.append(penalty)

        # sum losses and penalties across environments
        total_erm = torch.stack(env_erm_losses).sum()
        total_penalty = torch.stack(env_penalties).sum()

        # eq. (1) from https://arxiv.org/abs/1907.02893
        total_loss = total_erm + self.lambda_irm * total_penalty

        metrics = {
            "erm_loss": total_erm.detach(),
            "irm_penalty": total_penalty.detach(),
        }

        return total_loss, metrics


class IRMModel(PyanNet):
    def __init__(self, lambda_irm: float = 1e2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.irm_loss = IRMLoss(lambda_irm=lambda_irm)

    def training_step(
        self,
        batch: TrainingBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Override training step to use IRM loss"""
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

        env_ids = domains.detach().to(device=waveforms.device)

        # Compute IRM loss - IRMLoss handles the reshaping
        loss, metrics = self.irm_loss(outputs, speaker_truth, env_ids)

        # Log metrics
        self.log("train_loss", loss)
        self.log("train_erm", metrics["erm_loss"])
        self.log("train_penalty", metrics["irm_penalty"])

        return loss
