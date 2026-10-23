"""Cost-aware evaluation of multi-horizon quantile forecasts.

Directional accuracy and quantile coverage use every origin in the slice.
The simulated book is long-only and non-overlapping: a long is opened only
when the median simple return clears the full round-trip cost, held for
`horizon` trading days, then closed. Opening pays `COST_PER_SIDE` and
closing pays `COST_PER_SIDE`. Flat periods earn zero and pay nothing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from model.costs import ROUND_TRIP_COST


def evaluate_slices(
    *,
    origins: pd.Series,
    time_idx: np.ndarray,
    horizon: int,
    pred_log_q10: np.ndarray,
    pred_log_q50: np.ndarray,
    pred_log_q90: np.ndarray,
    actual_log: np.ndarray,
    masks: dict[str, np.ndarray],
    recent_limit: int = 8,
) -> dict:
    """Score one horizon on each named boolean mask (validation, OOS, ...)."""

    pred_log_q10 = np.asarray(pred_log_q10, dtype=np.float64)
    pred_log_q50 = np.asarray(pred_log_q50, dtype=np.float64)
    pred_log_q90 = np.asarray(pred_log_q90, dtype=np.float64)
    actual_log = np.asarray(actual_log, dtype=np.float64)
    time_idx = np.asarray(time_idx, dtype=np.int64)
    origin_dates = pd.to_datetime(origins)

    report: dict[str, dict] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if int(mask.sum()) == 0:
            report[name] = _empty_slice()
            continue
        report[name] = _score_slice(
            origin_dates=origin_dates[mask],
            time_idx=time_idx[mask],
            horizon=horizon,
            q10=pred_log_q10[mask],
            q50=pred_log_q50[mask],
            q90=pred_log_q90[mask],
            actual=actual_log[mask],
            recent_limit=recent_limit,
        )
    return report


def _score_slice(
    *,
    origin_dates: pd.Series,
    time_idx: np.ndarray,
    horizon: int,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    actual: np.ndarray,
    recent_limit: int,
) -> dict:
    order = np.argsort(time_idx, kind="mergesort")
    origin_dates = origin_dates.iloc[order].reset_index(drop=True)
    time_idx = time_idx[order]
    q10, q50, q90, actual = q10[order], q50[order], q90[order], actual[order]

    pred_simple = np.expm1(q50)
    actual_simple = np.expm1(actual)
    nonzero = actual != 0
    directional_hits = np.sign(q50[nonzero]) == np.sign(actual[nonzero])
    covered = (actual >= np.minimum(q10, q90)) & (actual <= np.maximum(q10, q90))

    trade = _non_overlapping_mask(time_idx, horizon)
    took_long = trade & (pred_simple > ROUND_TRIP_COST)
    # Round-trip: entry and exit each cost COST_PER_SIDE, summed as ROUND_TRIP_COST.
    net = np.zeros(len(q50), dtype=np.float64)
    net[took_long] = actual_simple[took_long] - ROUND_TRIP_COST
    traded_net = net[trade]
    equity = np.cumprod(1.0 + traded_net) if len(traded_net) else np.array([1.0])
    sharpe = _sharpe(traded_net, horizon)
    drawdown = _max_drawdown(equity)

    recent_rows = []
    recent_index = np.flatnonzero(trade)[-recent_limit:]
    for position in recent_index:
        recent_rows.append(
            {
                "origin": pd.Timestamp(origin_dates.iloc[position]).date().isoformat(),
                "predicted_simple": float(pred_simple[position]),
                "actual_simple": float(actual_simple[position]),
                "signal": "long" if bool(took_long[position]) else "flat",
                "net_simple": float(net[position]),
                "hit": bool(np.sign(q50[position]) == np.sign(actual[position]) and actual[position] != 0),
            }
        )

    wins = actual_simple[took_long] - ROUND_TRIP_COST > 0 if took_long.any() else np.array([])
    return {
        "n": int(len(q50)),
        "n_trades": int(took_long.sum()),
        "directional_accuracy": _mean(directional_hits),
        "quantile_coverage_10_90": _mean(covered),
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "cumulative_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "trade_win_rate": _mean(wins) if len(wins) else None,
        "recent": recent_rows,
    }


def _non_overlapping_mask(time_idx: np.ndarray, horizon: int) -> np.ndarray:
    take = np.zeros(len(time_idx), dtype=bool)
    next_free = -10**18
    for i, idx in enumerate(time_idx):
        if int(idx) >= next_free:
            take[i] = True
            next_free = int(idx) + horizon
    return take


def _sharpe(per_trade_simple: np.ndarray, horizon: int) -> float | None:
    if len(per_trade_simple) < 2:
        return None
    std = float(np.std(per_trade_simple, ddof=1))
    if std == 0 or not np.isfinite(std):
        return None
    periods_per_year = 252.0 / float(horizon)
    return float(np.mean(per_trade_simple) / std * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float | None:
    if len(equity) == 0:
        return None
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _mean(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _empty_slice() -> dict:
    return {
        "n": 0,
        "n_trades": 0,
        "directional_accuracy": None,
        "quantile_coverage_10_90": None,
        "sharpe": None,
        "max_drawdown": None,
        "cumulative_return": None,
        "trade_win_rate": None,
        "recent": [],
    }


def slice_masks(
    *,
    time_idx: np.ndarray,
    validation_start_idx: int,
    in_sample_end_idx: int,
    oos_start_idx: int | None,
) -> dict[str, np.ndarray]:
    """Classify each forecast origin. An origin is the first decoder session."""

    index = np.asarray(time_idx)
    masks = {
        "train_fit": index < validation_start_idx,
        "validation": (index >= validation_start_idx) & (index <= in_sample_end_idx),
        "out_of_sample": np.zeros(len(index), dtype=bool)
        if oos_start_idx is None
        else index >= oos_start_idx,
    }
    return masks


def as_date(value) -> date:
    return pd.Timestamp(value).date()
