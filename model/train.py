"""Train one TFT checkpoint and write the costed walk-forward report.

The network is the pytorch-forecasting TemporalFusionTransformer:

- Variable selection networks score static, encoder, and decoder covariates.
- Gated residual networks sit inside those selectors and around the LSTM.
- Interpretable multi-head attention reads the encoder time axis.
- QuantileLoss is the head. It emits 0.1, 0.5, and 0.9 on each future step.

`max_prediction_length` is 252 trading days. A one-day or one-month forecast
is the corresponding prefix of that path. Training uses only sessions through
the in-sample cutoff. The last 20% of the time index inside 2010–2023 is the
validation set used for early stopping. 2024 onward is scored later, with
frozen weights, and is not in the training or validation datasets.

Both backtests charge 10 bps commission plus 5 bps slippage on each side.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from model.config import (
    ATTENTION_HEAD_SIZE,
    COMMISSION_PER_SIDE,
    COST_PER_SIDE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    DROPOUT,
    HIDDEN_CONTINUOUS_SIZE,
    HIDDEN_SIZE,
    HORIZONS,
    LEARNING_RATE,
    QUANTILES,
    ROUND_TRIP_COST,
    SLIPPAGE_PER_SIDE,
    TRAIN_END,
    UNKNOWN_REALS,
    artifact_dir,
    normalize_ticker,
)
from model.dataset import SplitData, build_splits
from model.device import resolve_accelerator
from model.evaluate import run_backtests
from model.features import prepare_history
from model.progress import TrainingProgress

logger = logging.getLogger(__name__)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _vector(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    while array.ndim > 1:
        array = array.mean(axis=0)
    return array


def _named_weights(names: list[str], values) -> list[dict]:
    weights = np.abs(_vector(values))
    if len(names) != len(weights):
        names = [f"variable_{index}" for index in range(len(weights))]
    total = float(weights.sum()) or 1.0
    pairs = sorted(
        ({"name": name, "weight": float(weight / total)} for name, weight in zip(names, weights)),
        key=lambda item: item["weight"],
        reverse=True,
    )
    return pairs


def _attention_profile(values) -> list[dict]:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    array = np.asarray(array, dtype=np.float64)
    profile = array if array.ndim == 1 else array.mean(axis=tuple(range(array.ndim - 1)))
    profile = np.clip(profile, 0, None)
    total = float(profile.sum()) or 1.0
    profile = profile / total
    start = -(len(profile) - 1)
    return [{"offset": int(start + index), "weight": float(weight)} for index, weight in enumerate(profile)]


def _assert_decoder_contract(model) -> None:
    decoder_vars = set(getattr(model, "decoder_variables", []))
    if not decoder_vars:
        logger.warning("Checkpoint did not expose decoder_variables; skipping the leakage check.")
        return
    leaked = sorted(set(UNKNOWN_REALS) & decoder_vars)
    if leaked:
        raise RuntimeError(
            "Observed features were wired into the decoder and could be read as future inputs: "
            + ", ".join(leaked)
        )
    logger.info("Decoder covariates: %s", ", ".join(sorted(decoder_vars)))


def _interpretation(model, dataset, accelerator: str) -> dict:
    sample = dataset
    if len(dataset) > 32:
        latest = dataset.decoded_index["time_idx_first_prediction"].nlargest(32).min()
        sample = dataset.filter(lambda index: (index["time_idx_first_prediction"] >= latest).to_numpy())
    loader = sample.to_dataloader(train=False, batch_size=16, num_workers=0)
    raw = model.predict(
        loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
    output = raw.output if hasattr(raw, "output") else raw[0]
    interpretation = model.interpret_output(output, reduction="sum")
    payload = {
        "encoder_variables": _named_weights(list(model.encoder_variables), interpretation["encoder_variables"]),
        "decoder_variables": _named_weights(list(model.decoder_variables), interpretation["decoder_variables"]),
        "static_variables": _named_weights(list(model.static_variables), interpretation["static_variables"]),
        "attention": _attention_profile(interpretation["attention"]),
    }
    return payload


def _trainer(destination: Path, accelerator: str, max_epochs: int, smoke: bool, on_progress=None) -> Trainer:
    destination.mkdir(parents=True, exist_ok=True)
    planned = 1 if smoke else max_epochs
    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=str(destination),
            filename="tft",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]
    if on_progress is not None:
        callbacks.append(TrainingProgress(on_progress, planned))
    return Trainer(
        max_epochs=1 if smoke else max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(destination), name="lightning"),
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0 if smoke else 2,
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=2 if smoke else 1.0,
        precision="32-true",
        log_every_n_steps=1 if smoke else 10,
    )


def _fit(
    model,
    train_loader,
    val_loader,
    destination: Path,
    accelerator: str,
    max_epochs: int,
    smoke: bool,
    on_progress=None,
):
    trainer = _trainer(destination, accelerator, max_epochs, smoke, on_progress)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


def _build_model(training_dataset):
    # hidden_size is the GRN / LSTM width. attention_head_size is the number
    # of heads in the interpretable attention block. The loss is the quantile head.
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(quantiles=list(QUANTILES)),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )


def train_ticker(
    ticker: str = "AAPL",
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    accelerator: str = "auto",
    smoke: bool = False,
    on_progress=None,
) -> dict:
    ticker = normalize_ticker(ticker)
    planned = 1 if smoke else max_epochs

    def report(payload: dict) -> None:
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            logger.exception("Training progress report failed")

    report(
        {
            "phase": "preparing",
            "message": "Preparing price history",
            "epoch": 0,
            "max_epochs": planned,
            "progress": 0.02,
        }
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed_everything(42, workers=True)
    device = resolve_accelerator(accelerator)
    logger.info("Training %s on %s", ticker, device)

    bundle = prepare_history(ticker, refresh=True)
    splits: SplitData = build_splits(bundle.frame)
    logger.info(
        "Cutoff %s (%s). Validation %s → %s. OOS starts %s. Train samples %s, validation samples %s.",
        splits.training_cutoff,
        splits.train_end_date,
        splits.validation_start_date,
        splits.validation_end_date,
        splits.oos_start_date,
        len(splits.training),
        len(splits.validation),
    )

    train_loader = splits.training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
    )
    val_loader = splits.validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )
    model = _build_model(splits.training)
    _assert_decoder_contract(model)
    logger.info("TFT parameters: %.1fk", model.size() / 1e3)
    report(
        {
            "phase": "training",
            "message": f"Epoch 1 of {planned}",
            "epoch": 1,
            "max_epochs": planned,
            "progress": 0.08,
        }
    )

    folder = artifact_dir(ticker)
    work = folder / "work"
    if work.exists():
        shutil.rmtree(work)
    try:
        trainer = _fit(model, train_loader, val_loader, work, device, max_epochs, smoke, report)
    except RuntimeError as exc:
        if device == "cpu":
            raise
        logger.warning("Training on %s failed (%s). Rebuilding on CPU.", device, exc)
        report(
            {
                "phase": "training",
                "message": "Continuing on CPU",
                "epoch": 1,
                "max_epochs": planned,
                "progress": 0.08,
            }
        )
        if work.exists():
            shutil.rmtree(work)
        model = _build_model(splits.training)
        trainer = _fit(model, train_loader, val_loader, work, "cpu", max_epochs, smoke, report)
    best_path = trainer.checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("Training finished without a checkpoint.")
    trained_on = "mps" if "mps" in trainer.accelerator.__class__.__name__.lower() else "cpu"

    best = TemporalFusionTransformer.load_from_checkpoint(best_path, map_location="cpu")
    best.eval()
    checkpoint_path = folder / "tft.ckpt"
    shutil.copy(best_path, checkpoint_path)
    dataset_path = folder / "dataset.pt"
    torch.save(splits.training, dataset_path)

    horizons = (1,) if smoke else HORIZONS
    report(
        {
            "phase": "scoring",
            "message": "Scoring the forecast",
            "epoch": planned,
            "max_epochs": planned,
            "progress": 0.86,
        }
    )
    evaluation = run_backtests(
        best,
        splits.training,
        bundle.frame,
        training_cutoff=splits.training_cutoff,
        oos_start=splits.oos_start,
        accelerator="cpu",
        horizons=horizons,
        max_origins=16 if smoke else None,
    )
    try:
        interpretation = _interpretation(best, splits.validation, "cpu")
    except Exception as exc:
        logger.exception("Interpretability export failed")
        interpretation = {"error": str(exc), "encoder_variables": [], "decoder_variables": [], "static_variables": [], "attention": []}

    meta = {
        "ticker": ticker,
        "smoke": smoke,
        "train_end_boundary": TRAIN_END,
        "training_cutoff_time_idx": splits.training_cutoff,
        "train_end_date": splits.train_end_date,
        "validation_start_date": splits.validation_start_date,
        "validation_end_date": splits.validation_end_date,
        "oos_start_date": splits.oos_start_date,
        "max_encoder_length": 252,
        "min_encoder_length": 126,
        "max_prediction_length": 252,
        "quantiles": list(QUANTILES),
        "hidden_size": HIDDEN_SIZE,
        "attention_head_size": ATTENTION_HEAD_SIZE,
        "dropout": DROPOUT,
        "costs": {
            "commission_per_side": COMMISSION_PER_SIDE,
            "slippage_per_side": SLIPPAGE_PER_SIDE,
            "cost_per_side": COST_PER_SIDE,
            "round_trip": ROUND_TRIP_COST,
        },
        "accelerator": trained_on,
        "best_checkpoint": str(checkpoint_path),
    }
    report({"phase": "saving", "message": "Saving the checkpoint", "progress": 0.96})
    (folder / "meta.json").write_text(json.dumps(_json_ready(meta), indent=2))
    (folder / "evaluation.json").write_text(json.dumps(_json_ready(evaluation), indent=2))
    (folder / "interpretability.json").write_text(json.dumps(_json_ready(interpretation), indent=2))
    shutil.rmtree(work, ignore_errors=True)
    logger.info("Saved %s", checkpoint_path)
    return _json_ready({"meta": meta, "evaluation": evaluation, "interpretation": interpretation})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local multi-horizon TFT.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--accelerator", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and two batches, used to check the pipeline. Not a usable forecast model.",
    )
    args = parser.parse_args()
    train_ticker(
        ticker=args.ticker,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
