"""Report epoch and batch progress while a TFT is fitting.

The fraction leaves the first slice for data preparation and the last slice
for scoring, so the bar does not sit at 100% while the walk-forward is still
running. Early stopping ends training before `max_epochs`; the caller moves
the phase to scoring rather than stretching the leftover epochs.
"""

from __future__ import annotations

import logging
import math

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

TRAIN_START = 0.08
TRAIN_SPAN = 0.74


def training_fraction(epoch: int, max_epochs: int, batch: int, batches: int | None) -> float:
    """Map the current epoch and batch onto the training slice of the bar."""
    within = 0.0
    if batches and batches > 0:
        within = min(max(batch / batches, 0.0), 1.0)
    completed = ((max(epoch, 1) - 1) + within) / max(max_epochs, 1)
    return min(TRAIN_START + TRAIN_SPAN * completed, TRAIN_START + TRAIN_SPAN)


def _metric(metrics, *names: str) -> float | None:
    for name in names:
        value = metrics.get(name) if hasattr(metrics, "get") else None
        if value is None:
            continue
        number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if math.isfinite(number):
            return number
    return None


class TrainingProgress(Callback):
    """Push epoch, batch, and loss into a caller-supplied reporter."""

    def __init__(self, emit, max_epochs: int) -> None:
        super().__init__()
        self.emit = emit
        self.max_epochs = max(int(max_epochs), 1)

    def _send(self, trainer, epoch: int, batch: int | None, batches: int | None, *, epoch_done: bool = False) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        step = batches if epoch_done and batches else batch
        payload = {
            "phase": "training",
            "message": f"Epoch {epoch} of {self.max_epochs}",
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "batch": None if epoch_done else batch,
            "batches": None if epoch_done else batches,
            "train_loss": _metric(metrics, "train_loss_step", "train_loss_epoch", "train_loss"),
            "val_loss": _metric(metrics, "val_loss"),
            "progress": training_fraction(
                epoch,
                self.max_epochs,
                1 if epoch_done else (step or 0),
                1 if epoch_done else batches,
            ),
        }
        try:
            self.emit(payload)
        except Exception:
            logger.exception("Training progress report failed")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        total = getattr(trainer, "num_training_batches", None)
        batches = int(total) if isinstance(total, (int, float)) and math.isfinite(total) and total > 0 else None
        self._send(trainer, int(trainer.current_epoch) + 1, batch_idx + 1, batches)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        total = getattr(trainer, "num_training_batches", None)
        batches = int(total) if isinstance(total, (int, float)) and math.isfinite(total) and total > 0 else None
        self._send(trainer, int(trainer.current_epoch) + 1, batches, batches, epoch_done=True)
