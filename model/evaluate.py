"""Walk-forward backtests with the same costs in every split.

Directional accuracy and quantile coverage use every origin, because those
are properties of the forecast, not of a position. The Sharpe ratio and the
drawdown use non-overlapping holds of length `horizon`: a hold is entered
only when the median simple return exceeds the 30 bp round trip, and that
round trip is subtracted once. Overlapping positions would either double-count
costs or hide them.

Shorter horizons are the leading steps of an h-step decode. Origins that do
not yet have h realized sessions are left unscored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from model.config import (
    HORIZONS,
    QUANTILES,
    ROUND_TRIP_COST,
    TARGET,
    TIME_IDX,
    TRAIN_END,
)
from model.dataset import make_dataset
from model.inference import predict_quantiles
from model.intervals import cumulative_log_quantiles

Q10, Q50, Q90 = (QUANTILES.index(0.1), QUANTILES.index(0.5), QUANTILES.index(0.9))


def _one_origin_each(dataset: TimeSeriesDataSet, max_origins: int | None) -> TimeSeriesDataSet:
    """Keep a single window per decoder start.

    The dataset index also contains shorter copies that finish on the same
    day. Scoring all of them would count one session many times.
    """
    decoded = dataset.decoded_index.reset_index(drop=True)
    if decoded.empty:
        return dataset
    encoder_length = decoded["time_idx_first_prediction"] - decoded["time_idx_first"]
    ranked = decoded.assign(_encoder=encoder_length, _row=np.arange(len(decoded)))
    chosen = ranked.sort_values("_encoder", ascending=False).drop_duplicates(
        "time_idx_first_prediction",
        keep="first",
    )
    if max_origins is not None:
        chosen = chosen.nlargest(max_origins, "time_idx_first_prediction")
    keep = set(int(row) for row in chosen["_row"])
    mask = np.array([index in keep for index in range(len(decoded))], dtype=bool)
    return dataset.filter(lambda index: mask)


def _sharpe(returns: np.ndarray, horizon: int) -> float | None:
    if len(returns) < 2:
        return None
    volatility = float(np.std(returns, ddof=1))
    if not np.isfinite(volatility) or volatility < 1e-8:
        return None
    periods = 252 / horizon
    value = float(np.mean(returns) / volatility * np.sqrt(periods))
    return value if np.isfinite(value) else None


def _max_drawdown(returns: np.ndarray) -> float | None:
    if len(returns) == 0:
        return None
    equity = np.cumprod(1 + np.maximum(returns, -0.999))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def _score_horizon(
    quantiles: np.ndarray,
    time_idx: np.ndarray,
    actual_by_time: pd.Series,
    horizon: int,
) -> dict:
    order = np.argsort(time_idx[:, 0], kind="mergesort")
    quantiles = quantiles[order]
    time_idx = time_idx[order]
    _, unique_at = np.unique(time_idx[:, 0], return_index=True)
    quantiles = quantiles[unique_at, :horizon, :]
    time_idx = time_idx[unique_at, :horizon]
    actual = np.array(
        [[float(actual_by_time.get(int(step), np.nan)) for step in row] for row in time_idx],
        dtype=np.float64,
    )
    valid = np.isfinite(actual).all(axis=1) & np.isfinite(quantiles).all(axis=(1, 2))
    quantiles = quantiles[valid]
    actual = actual[valid]
    time_idx = time_idx[valid]
    if len(quantiles) == 0:
        return {
            "directional_accuracy": None,
            "quantile_coverage": None,
            "sharpe": None,
            "max_drawdown": None,
            "n_origins": 0,
            "n_trades": 0,
            "strategy_return": None,
        }

    log_q10, log_q50, log_q90 = cumulative_log_quantiles(quantiles, Q10, Q50, Q90)
    pred_q10 = log_q10[:, -1]
    pred_q50 = log_q50[:, -1]
    pred_q90 = log_q90[:, -1]
    actual_cum = actual.sum(axis=1)
    pred_simple = np.exp(pred_q50) - 1
    actual_simple = np.exp(actual_cum) - 1
    nonzero = actual_cum != 0
    directional = None
    if nonzero.any():
        directional = float(np.mean(np.sign(pred_q50[nonzero]) == np.sign(actual_cum[nonzero])))
    lower = np.minimum(pred_q10, pred_q90)
    upper = np.maximum(pred_q10, pred_q90)
    coverage = float(np.mean((actual_cum >= lower) & (actual_cum <= upper)))

    selected = np.arange(0, len(pred_simple), horizon)
    entered = pred_simple[selected] > ROUND_TRIP_COST
    trade_returns = np.where(entered, actual_simple[selected] - ROUND_TRIP_COST, 0.0)
    strategy_return = float(np.prod(1 + np.maximum(trade_returns, -0.999)) - 1)
    return {
        "directional_accuracy": directional,
        "quantile_coverage": coverage,
        "sharpe": _sharpe(trade_returns, horizon),
        "max_drawdown": _max_drawdown(trade_returns),
        "n_origins": int(len(pred_simple)),
        "n_trades": int(entered.sum()),
        "strategy_return": strategy_return,
    }


def _recent_rows(
    quantiles: np.ndarray,
    time_idx: np.ndarray,
    frame: pd.DataFrame,
    limit: int = 15,
) -> list[dict]:
    actual_by_time = frame.set_index(TIME_IDX)[TARGET]
    date_by_time = frame.set_index(TIME_IDX)["date"]
    rows: list[dict] = []
    for sample in range(quantiles.shape[0]):
        step = int(time_idx[sample, 0])
        if step not in actual_by_time.index:
            continue
        actual = float(actual_by_time.loc[step])
        q10 = float(quantiles[sample, 0, Q10])
        q50 = float(quantiles[sample, 0, Q50])
        q90 = float(quantiles[sample, 0, Q90])
        rows.append(
            {
                "date": str(pd.Timestamp(date_by_time.loc[step]).date()),
                "time_idx": step,
                "actual_log_return": actual,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "in_band": bool(min(q10, q90) <= actual <= max(q10, q90)),
            }
        )
    rows.sort(key=lambda row: row["time_idx"])
    deduped: dict[str, dict] = {}
    for row in rows:
        deduped[row["date"]] = row
    trimmed = list(deduped.values())[-limit:]
    for row in trimmed:
        row.pop("time_idx", None)
    return trimmed


def evaluate_split(
    model,
    training: TimeSeriesDataSet,
    frame: pd.DataFrame,
    *,
    min_prediction_idx: int,
    date_limit: pd.Timestamp | None,
    accelerator: str,
    horizons: tuple[int, ...] = HORIZONS,
    max_origins: int | None = None,
    collect_recent: bool = False,
) -> dict:
    scope = frame if date_limit is None else frame[frame["date"] <= date_limit]
    scope = scope.reset_index(drop=True)
    actual_by_time = scope.set_index(TIME_IDX)[TARGET]
    by_horizon: dict[str, dict] = {}
    recent: list[dict] = []
    for horizon in horizons:
        dataset = make_dataset(
            scope,
            training=training,
            min_prediction_idx=min_prediction_idx,
            max_prediction_length=horizon,
            min_prediction_length=horizon,
        )
        dataset = _one_origin_each(dataset, max_origins)
        quantiles, time_idx = predict_quantiles(model, dataset, accelerator=accelerator)
        by_horizon[str(horizon)] = _score_horizon(quantiles, time_idx, actual_by_time, horizon)
        if collect_recent and horizon == 1:
            recent = _recent_rows(quantiles, time_idx, scope)
    return {"by_horizon": by_horizon, "recent": recent}


def run_backtests(
    model,
    training: TimeSeriesDataSet,
    frame: pd.DataFrame,
    *,
    training_cutoff: int,
    oos_start: int | None,
    accelerator: str,
    horizons: tuple[int, ...] = HORIZONS,
    max_origins: int | None = None,
) -> dict:
    insample = evaluate_split(
        model,
        training,
        frame,
        min_prediction_idx=training_cutoff + 1,
        date_limit=pd.Timestamp(TRAIN_END),
        accelerator=accelerator,
        horizons=horizons,
        max_origins=max_origins,
        collect_recent=oos_start is None,
    )
    report: dict = {"in_sample": insample, "out_of_sample": None}
    if oos_start is not None:
        report["out_of_sample"] = evaluate_split(
            model,
            training,
            frame,
            min_prediction_idx=oos_start,
            date_limit=None,
            accelerator=accelerator,
            horizons=horizons,
            max_origins=max_origins,
            collect_recent=True,
        )
    return report
