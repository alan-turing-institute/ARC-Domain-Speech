from typing import Any, TypedDict

import torch
import torch.nn as nn

from dr_sad.pyannet import PyanNet


class TrainingBatch(TypedDict):
    waveforms: torch.Tensor
    annotations: list[list[tuple[float, float]]]
    domains: list[int]


class IRMModel(PyanNet):
    def __init__(
        self,
        lambda_irm: float,
        lambda_scheduling_steps: int | None = None,
        lambda_start_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.lambda_scheduling_steps = lambda_scheduling_steps
        self.target_lambda = float(lambda_irm)

        # Type annotations (appeases mypy)
        self.anneal_step: int | None
        self.lambda_start_step: int | None

        if lambda_scheduling_steps is not None:
            self.lambda_irm = 0.0  # Start at 0
            self.anneal_step = 0
            self.lambda_start_step = (
                lambda_start_step if lambda_start_step is not None else 0
            )
        else:
            self.lambda_irm = self.target_lambda  # Use target immediately
            self.anneal_step = None
            self.lambda_start_step = None

        self.dummy_w = nn.Parameter(torch.tensor(1.0))

    def configure_optimizers(self):
        """Override to exclude dummy_w from optimization"""
        # Get all parameters except dummy_w
        params_to_optimize = [
            p for name, p in self.named_parameters() if name != "dummy_w"
        ]

        # Use parent's optimizer settings but with filtered parameters
        return torch.optim.Adam(params_to_optimize, lr=self.hparams.learning_rate)

    def step_linear_lambda_scheduler(self) -> None:
        """Linearly increase lambda_irm over the specified number of steps"""

        if (
            self.anneal_step is not None
            and self.lambda_scheduling_steps is not None
            and self.lambda_start_step is not None
        ):
            # Wait until we reach the start step
            if self.anneal_step < self.lambda_start_step:
                self.lambda_irm = 0.0
                self.anneal_step += 1
            # Linear scheduling phase
            elif (
                self.anneal_step < self.lambda_start_step + self.lambda_scheduling_steps
            ):
                steps_since_start = self.anneal_step - self.lambda_start_step
                self.lambda_irm = float(
                    self.target_lambda
                    * (steps_since_start / self.lambda_scheduling_steps)
                )
                self.anneal_step += 1
            # After scheduling is complete
            else:
                self.lambda_irm = float(self.target_lambda)
        else:
            self.lambda_irm = float(self.target_lambda)

    def IRMLoss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Implements equation (1) from https://arxiv.org/abs/1907.02893
        Args:
            logits: (batch, channels, frames) - as output from PyanNet
            labels: (batch, channels, frames) - as output from prepare_annotation
            env_ids: (batch,) - environment ID for each sample

        Returns:
            loss: scalar
            metrics: dict with 'erm_loss' and 'irm_penalty'
        """
        unique_envs = env_ids.unique()

        # Collect per-environment losses to avoid inefficient tensor accumulation
        env_erm_losses = []
        env_penalties = []

        # Compute ERM loss and IRM penalty for each environment to capture
        # per-environment behavior
        for env in unique_envs:
            # Get samples from this environment
            mask = env_ids == env
            env_logits = logits[mask]  # (batch, channels, frames)
            env_labels = labels[mask]  # (batch, frames)
            _domains = env_ids[mask]

            # Scale logits by dummy classifier
            env_logits_scaled = env_logits * self.dummy_w

            # Compute loss on logits (this is R^e(w·Φ)) in eq. (1)
            erm_loss = self.loss_function(
                env_labels, _domains.tolist(), env_logits_scaled
            )
            env_erm_losses.append(erm_loss)

            # IRM penalty: gradient of the loss w.r.t. dummy_w
            grad = torch.autograd.grad(
                erm_loss, self.dummy_w, create_graph=True, retain_graph=True
            )[0]
            penalty = grad**2
            env_penalties.append(penalty)

        # Sum losses and penalties across environments: Σ_e [...]
        total_erm = torch.stack(env_erm_losses).sum()
        total_penalty = torch.stack(env_penalties).sum()

        # Equation (1): L_IRM = Σ_e R^e(w∘Φ) + λ·Σ_e ||∇_w R^e(w∘Φ)||²
        total_loss = total_erm + self.lambda_irm * total_penalty

        metrics = {
            "erm_loss": total_erm.detach(),
            "irm_penalty": total_penalty.detach(),
        }

        return total_loss, metrics

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
        # Update lambda_irm if using scheduling
        if self.anneal_step is not None:
            self.step_linear_lambda_scheduler()

        # Forward pass
        outputs = self(waveforms)
        # (batch, time, channels) -> (batch, channels, time)
        outputs = outputs.swapaxes(1, 2)

        # Prepare ground truth
        speaker_truth = self.prepare_annotation(waveforms, annotations)

        # Compute IRM loss - IRMLoss handles the reshaping
        loss, metrics = self.IRMLoss(outputs, speaker_truth, domains)

        # Log metrics
        self.log("train_loss", loss)
        self.log("train_erm", metrics["erm_loss"])
        self.log("train_irm_penalty", metrics["irm_penalty"])
        self.log("lambda_irm", self.lambda_irm)

        return loss

    def evaluate_batch_irm(
        self, batch: Any, _batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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

        with torch.set_grad_enabled(True):  # need grad for IRM penalty
            total_loss, metrics = self.IRMLoss(outputs, speaker_truth, _domains)

        erm_loss, irm_penalty = metrics["erm_loss"], metrics["irm_penalty"]
        accuracy = self.accuracy_function(speaker_truth, _domains, outputs)
        return total_loss, erm_loss, irm_penalty, accuracy

    def test_step(self, batch: Any, batch_idx: int) -> None:
        total_loss, accuracy = self.evaluate_batch(batch, batch_idx)
        self.log("test_loss", total_loss)
        self.log("test_accuracy", accuracy)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        total_loss, erm_loss, irm_penalty, accuracy = self.evaluate_batch_irm(
            batch, batch_idx
        )
        self.log("val_loss", total_loss)
        self.log("val_erm", erm_loss)
        self.log("val_irm_penalty", irm_penalty)
        self.log("val_accuracy", accuracy)
