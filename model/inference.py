"""Run a fitted TFT and return inverse-transformed quantile paths."""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet

from model.config import QUANTILES

logger = logging.getLogger(__name__)

# Lightning on MPS aborts the process inside the Metal driver
# (`MTLCommandBuffer` assertion). Python cannot catch that, and two
# overlapping forecasts trip it. One window is cheap on CPU.
_PREDICT_LOCK = threading.Lock()


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _quantiles_from_output(output) -> np.ndarray:
    tensor = output
    if hasattr(output, "prediction"):
        tensor = output.prediction
    elif isinstance(output, dict) and "prediction" in output:
        tensor = output["prediction"]
    array = _to_numpy(tensor)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise RuntimeError(f"Expected quantile predictions shaped (n, horizon, q), got {array.shape}.")
    return array.astype(np.float64)


def _context_from_result(result):
    if hasattr(result, "x") and isinstance(result.x, dict):
        return result.output if hasattr(result, "output") else result, result.x
    if isinstance(result, tuple):
        output = result[0]
        context = None
        for item in result[1:]:
            if isinstance(item, dict) and "decoder_time_idx" in item:
                context = item
                break
        return output, context
    return result, None


def predict_quantiles(
    model,
    dataset: TimeSeriesDataSet,
    accelerator: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (quantiles, decoder_time_idx).

    Quantiles are in log-return units, already passed back through the
    training GroupNormalizer. Axis 2 follows `model.loss.quantiles`.
    """
    if len(dataset) == 0:
        raise RuntimeError("Prediction dataset is empty.")
    if accelerator not in {None, "cpu"}:
        logger.info("Forecast runs on CPU. MPS predict can abort the server.")
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    with _PREDICT_LOCK:
        result = _predict(model, loader, "cpu")
    output, context = _context_from_result(result)
    quantiles = _quantiles_from_output(output)
    if context is None or "decoder_time_idx" not in context:
        raise RuntimeError("TFT prediction did not return decoder_time_idx.")
    time_idx = _to_numpy(context["decoder_time_idx"]).astype(np.int64)
    if time_idx.shape[0] != quantiles.shape[0] or time_idx.shape[1] != quantiles.shape[1]:
        raise RuntimeError(
            f"Time index shape {time_idx.shape} does not match predictions {quantiles.shape}."
        )
    expected = tuple(float(q) for q in getattr(model.loss, "quantiles", QUANTILES))
    if quantiles.shape[-1] != len(expected):
        raise RuntimeError(f"Model emitted {quantiles.shape[-1]} quantiles, expected {expected}.")
    return quantiles, time_idx


def _predict(model, loader, accelerator: str):
    return model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        trainer_kwargs={
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )
