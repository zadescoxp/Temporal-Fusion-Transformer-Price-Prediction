"""Build the latest encoder window and turn quantile log returns into prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.config import (
    COMMISSION_PER_SIDE,
    COST_PER_SIDE,
    HORIZON_LABELS,
    MAX_PREDICTION_LENGTH,
    ROUND_TRIP_COST,
    SLIPPAGE_PER_SIDE,
    normalize_ticker,
)
from model.dataset import make_dataset
from model.features import append_future, prepare_history
from model.inference import predict_quantiles
from model.intervals import cumulative_log_quantiles

from app.core.model_loader import load_model

_COSTS = {
    "commission_per_side": COMMISSION_PER_SIDE,
    "slippage_per_side": SLIPPAGE_PER_SIDE,
    "cost_per_side": COST_PER_SIDE,
    "round_trip": ROUND_TRIP_COST,
}


def _quantile_index(model) -> dict[float, int]:
    values = [float(item) for item in model.loss.quantiles]
    return {value: index for index, value in enumerate(values)}


def forecast_payload(ticker: str, horizons: list[int]) -> dict:
    ticker = normalize_ticker(ticker)
    loaded = load_model(ticker)
    bundle = prepare_history(ticker, refresh=False)
    extended = append_future(bundle, MAX_PREDICTION_LENGTH)
    dataset = make_dataset(
        extended,
        training=loaded.dataset,
        predict=True,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
    )
    quantiles, time_idx = predict_quantiles(
        loaded.model,
        dataset,
        accelerator="cpu",
        batch_size=1,
    )
    daily = quantiles[0]
    steps = time_idx[0]
    index = _quantile_index(loaded.model)
    log_q10, log_q50, log_q90 = cumulative_log_quantiles(daily, index[0.1], index[0.5], index[0.9])
    last = bundle.frame.iloc[-1]
    last_close = float(last.close)
    price_q10 = last_close * np.exp(log_q10)
    price_q50 = last_close * np.exp(log_q50)
    price_q90 = last_close * np.exp(log_q90)
    dates = {
        int(row.time_idx): pd.Timestamp(row.date).strftime("%Y-%m-%d")
        for row in extended[["time_idx", "date"]].itertuples(index=False)
    }

    path = []
    for step_index, step in enumerate(steps):
        day = dates.get(int(step))
        if day is None:
            raise RuntimeError(f"Prediction time index {int(step)} is not on the calendar frame.")
        path.append(
            {
                "date": day,
                "log_q10": float(log_q10[step_index]),
                "log_q50": float(log_q50[step_index]),
                "log_q90": float(log_q90[step_index]),
                "q10": float(price_q10[step_index]),
                "q50": float(price_q50[step_index]),
                "q90": float(price_q90[step_index]),
            }
        )

    cards = []
    for horizon in horizons:
        point = path[horizon - 1]
        median_simple = float(np.exp(point["log_q50"]) - 1)
        edge = median_simple - ROUND_TRIP_COST
        cards.append(
            {
                "horizon": horizon,
                "label": HORIZON_LABELS[horizon],
                "date": point["date"],
                "log_return": {
                    "q10": point["log_q10"],
                    "q50": point["log_q50"],
                    "q90": point["log_q90"],
                },
                "simple_return": {
                    "q10": float(np.exp(point["log_q10"]) - 1),
                    "q50": median_simple,
                    "q90": float(np.exp(point["log_q90"]) - 1),
                },
                "price": {"q10": point["q10"], "q50": point["q50"], "q90": point["q90"]},
                "signal": "long" if edge > 0 else "flat",
                "edge_after_cost": edge,
            }
        )

    return {
        "ticker": ticker,
        "as_of": pd.Timestamp(last.date).strftime("%Y-%m-%d"),
        "last_close": last_close,
        "smoke": bool(loaded.meta.get("smoke", False)),
        "costs": _COSTS,
        "horizons": cards,
        "path": path,
        "price_band_note": (
            "The median compounds the daily 0.5 path. "
            "The 0.1–0.9 band is the range for that cumulative move, so it widens with the square root of the horizon."
        ),
    }
