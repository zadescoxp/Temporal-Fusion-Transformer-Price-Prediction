"""Causal and scale checks for the indicator frame. No network and no download."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.config import UNKNOWN_REALS
from model.features import _wma, build_model_frame


def _ohlcv(n: int = 900, seed: int = 7, bump_tail: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    returns = rng.normal(0.0004, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    if bump_tail:
        close[-bump_tail:] *= 1.75
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 3_000_000, size=n),
        }
    )


class FeatureTests(unittest.TestCase):
    def test_wma_weights_the_newest_sample_most(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        weighted = _wma(series, 3)
        self.assertTrue(np.isnan(weighted.iloc[0]))
        self.assertTrue(np.isnan(weighted.iloc[1]))
        self.assertAlmostEqual(float(weighted.iloc[2]), (1 * 1 + 2 * 2 + 3 * 3) / 6)
        self.assertAlmostEqual(float(weighted.iloc[3]), (2 * 1 + 3 * 2 + 4 * 3) / 6)

    def test_future_prices_do_not_change_past_features(self) -> None:
        base = build_model_frame(_ohlcv(), "AAPL", set(), set())
        bumped = build_model_frame(_ohlcv(bump_tail=15), "AAPL", set(), set())
        guard = 200 + 15
        columns = list(UNKNOWN_REALS)
        left = base.iloc[:-guard][columns].reset_index(drop=True)
        right = bumped.iloc[:-guard][columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_indicators_are_normalized_before_the_model(self) -> None:
        frame = build_model_frame(_ohlcv(), "AAPL", set(), set())
        self.assertGreater(float(frame["close"].median()), 10)
        self.assertLess(float(frame["sma_200"].abs().median()), 1)
        self.assertLess(float(frame["ema_200"].abs().median()), 1)
        self.assertLess(float(frame["vwma_20"].abs().median()), 1)
        self.assertLess(float(frame["hull_9"].abs().median()), 1)
        self.assertLess(float(frame["ichimoku_base"].abs().median()), 1)
        self.assertTrue(frame["rsi_14"].between(0, 1).all())
        self.assertTrue(frame["stoch_k"].between(0, 1).all())
        self.assertTrue(frame["stoch_d"].between(0, 1).all())
        self.assertTrue(frame["adx_14"].between(0, 1).all())
        self.assertTrue(frame["williams_r"].between(-1, 0).all())
        self.assertTrue(frame["ultimate_osc"].between(0, 1).all())
        self.assertTrue(frame["stoch_rsi_k"].between(0, 1).all())
        self.assertLess(float(frame["macd"].abs().median()), 1)
        self.assertLess(float(frame["ao"].abs().median()), 1)
        self.assertLess(float(frame["momentum_10"].abs().median()), 1)
        self.assertLess(float(frame["bull_power"].abs().median()), 1)
        self.assertLess(float(frame["bear_power"].abs().median()), 1)


if __name__ == "__main__":
    unittest.main()
