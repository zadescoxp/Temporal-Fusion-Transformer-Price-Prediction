"""Cumulative price bands from per-step daily log-return quantiles.

The network predicts the log return of each future session, not the return
from today to that session. Summing the 0.1 path (or the 0.9 path) treats
every future session as an extreme day at the same time. A year of that
collapses the lower price toward zero and sends the upper price into the
hundreds of thousands.

The median cumulative log return is the sum of the daily medians. The band
is the root-sum-square of the daily deviations from that median, split into
downside and upside so a skewed daily head stays skewed. One session is left
unchanged: the root-sum-square of a single day is the day itself.
"""

from __future__ import annotations

import numpy as np

# Inverse CDF of the standard normal at 0.9. The 0.1 quantile is the negation.
Z_90 = 1.2815515655446004


def cumulative_log_quantiles(
    daily: np.ndarray,
    q10: int,
    q50: int,
    q90: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative log q10, q50, q90.

    `daily` is (..., steps, quantiles). The three results drop the quantile
    axis and keep the step axis, so index -1 is the full horizon.
    """
    steps = np.asarray(daily, dtype=np.float64)
    low = steps[..., q10]
    mid = steps[..., q50]
    high = steps[..., q90]
    down = np.maximum(mid - low, 0.0) / Z_90
    up = np.maximum(high - mid, 0.0) / Z_90
    cumulative_mid = np.cumsum(mid, axis=-1)
    cumulative_down = np.sqrt(np.cumsum(np.square(down), axis=-1))
    cumulative_up = np.sqrt(np.cumsum(np.square(up), axis=-1))
    return (
        cumulative_mid - Z_90 * cumulative_down,
        cumulative_mid,
        cumulative_mid + Z_90 * cumulative_up,
    )
