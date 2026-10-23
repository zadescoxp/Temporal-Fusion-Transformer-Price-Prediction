"""Load a ticker checkpoint once and keep it in memory."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from model.config import MODELS_DIR, normalize_ticker
from model.device import resolve_accelerator
from model.errors import ModelNotFoundError

_LOCK = threading.Lock()
_CACHE: dict[str, "LoadedModel"] = {}


@dataclass
class LoadedModel:
    ticker: str
    model: TemporalFusionTransformer
    dataset: TimeSeriesDataSet
    meta: dict
    accelerator: str
    folder: Path


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def artifact_folder(ticker: str) -> Path:
    return MODELS_DIR / normalize_ticker(ticker).lower()


def load_model(ticker: str) -> LoadedModel:
    ticker = normalize_ticker(ticker)
    with _LOCK:
        cached = _CACHE.get(ticker)
        if cached is not None:
            return cached
        folder = artifact_folder(ticker)
        checkpoint = folder / "tft.ckpt"
        dataset_path = folder / "dataset.pt"
        if not checkpoint.exists() or not dataset_path.exists():
            raise ModelNotFoundError(
                f"No trained model for {ticker}. Train it from the dashboard or run "
                f"`python -m model.train --ticker {ticker}`."
            )
        model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), map_location="cpu")
        model.eval()
        dataset = _torch_load(dataset_path)
        if not isinstance(dataset, TimeSeriesDataSet):
            raise ModelNotFoundError(f"Saved dataset for {ticker} is not a TimeSeriesDataSet.")
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        requested = str(meta.get("accelerator", "auto"))
        try:
            accelerator = resolve_accelerator(requested if requested in {"cpu", "mps", "auto"} else "auto")
        except Exception:
            accelerator = "cpu"
        loaded = LoadedModel(ticker, model, dataset, meta, accelerator, folder)
        _CACHE[ticker] = loaded
        return loaded


def invalidate(ticker: str) -> None:
    with _LOCK:
        _CACHE.pop(normalize_ticker(ticker), None)


def read_json(ticker: str, name: str) -> dict:
    path = artifact_folder(ticker) / name
    if not path.exists():
        raise ModelNotFoundError(
            f"No {name} for {normalize_ticker(ticker)}. Train the model to generate it."
        )
    return json.loads(path.read_text())
