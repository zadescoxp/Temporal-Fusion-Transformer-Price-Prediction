"""Load a trained TFT and turn its quantile head into log-return paths.

The checkpoint's dataset parameters carry the training-set normalizer.
Inference never refits it. `mode="prediction"` asks pytorch-forecasting to
invert that normalizer, so the tensors below are daily log returns.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.dataset import HORIZONS, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH
from model.intervals import cumulative_log_quantiles
from model.features import (
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    STATIC_CATEGORICALS,
    TARGET,
    UNKNOWN_REALS,
    extend_with_future_calendar,
)

log = logging.getLogger(__name__)


def read_parameters(path: Path) -> dict:
    # The dataset object contains fitted scalers, so this is a trusted local artifact.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "get_parameters"):
        return payload.get_parameters()
    raise TypeError(f"Unrecognized dataset artifact at {path}.")


def quantile_index(model) -> dict[float, int]:
    quantiles = [float(value) for value in model.loss.quantiles]
    return {value: quantiles.index(value) for value in (0.1, 0.5, 0.9)}


def predict_raw(model, dataset):
    """One forward pass over `dataset`. Weights stay frozen (`model.eval()`)."""

    model.eval()
    loader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0, shuffle=False)
    return model.predict(
        loader,
        mode="raw",
        return_x=True,
        return_index=True,
        trainer_kwargs={
            "logger": False,
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "accelerator": "cpu",
            "devices": 1,
        },
    )


def model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns the dataset is allowed to see. OHLC stays out so placeholder future prices cannot become inputs."""

    columns = list(
        dict.fromkeys(
            [
                "time_idx",
                *STATIC_CATEGORICALS,
                TARGET,
                *UNKNOWN_REALS,
                *KNOWN_CATEGORICALS,
                *KNOWN_REALS,
            ]
        )
    )
    return frame[columns].copy()


def step_log_returns(model, raw) -> np.ndarray:
    """Per-step quantiles in log-return space, shape (samples, steps, quantiles).

    The TFT forward pass already inverts the training-set target normalizer
    inside `transform_output`, so this tensor is not a z-score.
    """

    output = raw.output
    prediction = output["prediction"] if not hasattr(output, "prediction") else output.prediction
    if not torch.is_tensor(prediction):
        raise TypeError("TFT raw output has no prediction tensor.")
    del model
    return prediction.detach().cpu().numpy().astype(np.float64)


def latest_forecast(model, parameters, history: pd.DataFrame, earnings: set[date], dividends: set[date]) -> dict:
    """Forecast the next 252 sessions from the last complete bar."""

    from pytorch_forecasting import TimeSeriesDataSet

    extended = extend_with_future_calendar(
        history,
        MAX_PREDICTION_LENGTH,
        earnings_dates=earnings,
        dividend_dates=dividends,
    )
    window = model_columns(extended.iloc[-(MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH) :]).reset_index(drop=True)
    dataset = TimeSeriesDataSet.from_parameters(
        parameters,
        window,
        predict=True,
        stop_randomization=True,
    )
    raw = predict_raw(model, dataset)
    steps = step_log_returns(model, raw)
    if steps.shape[0] != 1:
        raise RuntimeError(f"Expected one forecast window, got {steps.shape[0]}.")
    future = extended.iloc[-MAX_PREDICTION_LENGTH :]
    return {
        "as_of": pd.Timestamp(history.iloc[-1]["date"]).date(),
        "last_close": float(history.iloc[-1]["close"]),
        "step_log": steps[0],
        "future_dates": [pd.Timestamp(value).date() for value in future["date"]],
        "quantiles": [float(value) for value in model.loss.quantiles],
        "interpretation": interpret(model, raw),
    }


def historical_origins(
    model,
    parameters,
    history: pd.DataFrame,
    earnings: set[date],
    dividends: set[date],
) -> pd.DataFrame:
    """Walk-forward quantile paths for every origin the frozen model can score.

    The frame is extended by a known-future calendar so origins near the
    present still have a 252-step decoder. Actuals are read only from real
    sessions. A horizon is left missing when its path runs off the realized sample.
    """

    from pytorch_forecasting import TimeSeriesDataSet

    extended = extend_with_future_calendar(
        history,
        MAX_PREDICTION_LENGTH,
        earnings_dates=earnings,
        dividend_dates=dividends,
    )
    dataset = TimeSeriesDataSet.from_parameters(
        parameters,
        model_columns(extended),
        stop_randomization=True,
    )
    raw = predict_raw(model, dataset)
    steps = step_log_returns(model, raw)
    decoder_index = raw.x["decoder_time_idx"].detach().cpu().numpy().astype(np.int64)
    if decoder_index.shape != steps.shape[:2]:
        raise RuntimeError(
            f"Decoder index shape {decoder_index.shape} does not match predictions {steps.shape[:2]}."
        )
    q = quantile_index(model)
    actual = {
        int(idx): float(value)
        for idx, value in zip(history["time_idx"].astype(int), history["log_return"].astype(float))
    }
    date_of = {
        int(idx): pd.Timestamp(value).date()
        for idx, value in zip(history["time_idx"].astype(int), history["date"])
    }
    rows: list[dict] = []
    for sample in range(steps.shape[0]):
        origin_idx = int(decoder_index[sample, 0])
        if origin_idx not in date_of:
            # Decoder starts on a synthetic session. There is no realized origin.
            continue
        row: dict = {"origin_time_idx": origin_idx, "origin_date": date_of[origin_idx]}
        for horizon in HORIZONS:
            slice_idx = decoder_index[sample, :horizon]
            realized = [actual.get(int(idx)) for idx in slice_idx]
            if any(value is None for value in realized):
                row[f"h{horizon}_q10"] = np.nan
                row[f"h{horizon}_q50"] = np.nan
                row[f"h{horizon}_q90"] = np.nan
                row[f"h{horizon}_actual"] = np.nan
                continue
            low, mid, high = cumulative_log_quantiles(
                steps[sample, :horizon],
                q[0.1],
                q[0.5],
                q[0.9],
            )
            row[f"h{horizon}_q10"] = float(low[-1])
            row[f"h{horizon}_q50"] = float(mid[-1])
            row[f"h{horizon}_q90"] = float(high[-1])
            row[f"h{horizon}_actual"] = float(sum(realized))
        rows.append(row)
    if not rows:
        raise RuntimeError("Walk-forward evaluation produced no origins.")
    return pd.DataFrame(rows).sort_values("origin_time_idx").reset_index(drop=True)


def interpret(model, raw) -> dict:
    """Variable-selection weights and encoder attention for the current batch.

    Variable selection (VSN) scores how much each static, encoder, and decoder
    input contributed. Encoder attention is the interpretable multi-head
    attention over past timesteps, averaged across the batch.
    """

    interpretation = model.interpret_output(raw.output, reduction="sum")
    static_names = list(model.static_variables)
    encoder_names = list(model.encoder_variables)
    decoder_names = list(model.decoder_variables)
    attention = interpretation["attention"]
    if torch.is_tensor(attention) and attention.ndim == 2:
        attention = attention.mean(dim=0)
    weights = attention.detach().cpu().numpy().astype(np.float64)
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    tail = [
        {"lag": int(len(weights) - 1 - index), "weight": float(weights[index])}
        for index in range(len(weights))
    ]
    # lag 0 is the most recent encoder session.
    tail = sorted(tail, key=lambda item: item["lag"])
    return {
        "static_variables": _importance(static_names, interpretation["static_variables"]),
        "encoder_variables": _importance(encoder_names, interpretation["encoder_variables"]),
        "decoder_variables": _importance(decoder_names, interpretation["decoder_variables"]),
        "attention": tail,
    }


def _importance(names: list[str], values) -> list[dict]:
    array = np.asarray(values.detach().cpu().numpy() if torch.is_tensor(values) else values, dtype=np.float64)
    array = np.atleast_1d(array).reshape(-1)
    if len(array) != len(names):
        names = [f"variable_{index}" for index in range(len(array))]
    total = float(array.sum())
    if total > 0:
        array = array / total
    order = np.argsort(array)[::-1]
    return [{"name": names[int(index)], "importance": float(array[int(index)])} for index in order]
