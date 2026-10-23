"""Methodology checks that do not require a trained network."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from model.backtest import evaluate_slices
from model.costs import ROUND_TRIP_COST
from model.features import (
    MODEL_COLUMNS,
    _post_holiday_flags,
    engineer_features,
    extend_with_future_calendar,
    nyse_holidays,
)
from model.splits import split_bounds


def _ohlcv(n: int = 420, seed: int = 7, bump_tail: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    returns = rng.normal(0.0002, 0.01, size=n)
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
    def test_future_prices_do_not_change_past_features(self) -> None:
        base = engineer_features(_ohlcv(), ticker="AAPL")
        bumped = engineer_features(_ohlcv(bump_tail=15), ticker="AAPL")
        # 60-session vol plus the 15 perturbed bars. Everything earlier must match.
        cutoff = pd.Timestamp(base["date"].iloc[-1]) - pd.tseries.offsets.BDay(15 + 60)
        columns = MODEL_COLUMNS + ["is_earnings_day", "is_dividend_day", "is_holiday"]
        left = base.loc[base["date"] <= cutoff, columns].reset_index(drop=True)
        right = bumped.loc[bumped["date"] <= cutoff, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_event_flags_only_on_the_event_session(self) -> None:
        raw = _ohlcv()
        event = pd.Timestamp(raw["date"].iloc[200]).date()
        dividend = pd.Timestamp(raw["date"].iloc[240]).date()
        frame = engineer_features(
            raw,
            ticker="AAPL",
            earnings_dates={event},
            dividend_dates={dividend},
        )
        earnings_hits = frame.loc[frame["is_earnings_day"] == 1, "date"]
        dividend_hits = frame.loc[frame["is_dividend_day"] == 1, "date"]
        self.assertEqual([pd.Timestamp(value).date() for value in earnings_hits], [event])
        self.assertEqual([pd.Timestamp(value).date() for value in dividend_hits], [dividend])

    def test_future_calendar_does_not_rewrite_history(self) -> None:
        frame = engineer_features(_ohlcv(), ticker="AAPL")
        future_event = pd.Timestamp(frame["date"].iloc[-1]).date() + timedelta(days=10)
        extended = extend_with_future_calendar(
            frame,
            20,
            earnings_dates={future_event},
        )
        self.assertEqual(len(extended), len(frame) + 20)
        np.testing.assert_allclose(extended.loc[: len(frame) - 1, "log_return"], frame["log_return"])
        self.assertTrue((extended.loc[: len(frame) - 1, "is_earnings_day"] == frame["is_earnings_day"]).all())
        self.assertEqual(int(extended["time_idx"].iloc[-1]), int(frame["time_idx"].iloc[-1]) + 20)
        self.assertTrue((extended.iloc[-20:]["log_return"] == 0).all())

    def test_post_holiday_flag_is_known_from_the_calendar(self) -> None:
        holidays = nyse_holidays(2023, 2024)
        self.assertIn(date(2024, 1, 1), holidays)
        self.assertIn(date(2024, 7, 4), holidays)
        sessions = [
            date(2023, 12, 29),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 7, 3),
            date(2024, 7, 5),
            date(2024, 7, 8),
        ]
        flags = _post_holiday_flags(sessions, holidays)
        self.assertEqual(flags.tolist(), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])


class SplitTests(unittest.TestCase):
    def test_cutoffs_are_chronological(self) -> None:
        dates = pd.bdate_range("2010-01-04", "2026-06-30")
        frame = pd.DataFrame({"date": dates, "time_idx": np.arange(len(dates))})
        bounds = split_bounds(frame)
        self.assertLess(bounds["training_cutoff"], bounds["validation_start_idx"])
        self.assertLessEqual(bounds["validation_start_date"], date(2023, 12, 31))
        self.assertLessEqual(bounds["training_cutoff_date"], bounds["validation_start_date"])
        self.assertGreaterEqual(bounds["oos_start_date"], date(2024, 1, 1))
        self.assertLess(bounds["in_sample_end_date"], bounds["oos_start_date"])
        span = bounds["in_sample_end_idx"] - int(frame["time_idx"].iloc[0])
        expected = int(frame["time_idx"].iloc[0]) + int(0.8 * span)
        self.assertEqual(bounds["validation_start_idx"], expected)


class CostTests(unittest.TestCase):
    def test_round_trip_is_deducted_only_when_the_signal_clears_it(self) -> None:
        # Three non-overlapping 1-day origins. The third forecast does not clear 30 bps.
        pred = np.array([0.010, 0.020, 0.001])
        actual = np.array([0.020, -0.010, 0.030])
        actual_log = np.log1p(actual)
        report = evaluate_slices(
            origins=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])),
            time_idx=np.array([0, 1, 2]),
            horizon=1,
            pred_log_q10=actual_log - 0.05,
            pred_log_q50=np.log1p(pred),
            pred_log_q90=actual_log + 0.05,
            actual_log=actual_log,
            masks={"out_of_sample": np.array([True, True, True])},
        )["out_of_sample"]
        self.assertEqual(report["n_trades"], 2)
        nets = [row["net_simple"] for row in report["recent"]]
        self.assertAlmostEqual(nets[0], actual[0] - ROUND_TRIP_COST)
        self.assertAlmostEqual(nets[1], actual[1] - ROUND_TRIP_COST)
        self.assertAlmostEqual(nets[2], 0.0)
        self.assertEqual([row["signal"] for row in report["recent"]], ["long", "long", "flat"])
        self.assertAlmostEqual(report["directional_accuracy"], 2 / 3)
        self.assertAlmostEqual(report["quantile_coverage_10_90"], 1.0)

    def test_flat_periods_stay_in_the_equity_curve(self) -> None:
        pred = np.array([0.0, 0.0])
        actual = np.array([0.05, -0.05])
        report = evaluate_slices(
            origins=pd.Series(pd.to_datetime(["2024-02-01", "2024-02-02"])),
            time_idx=np.array([10, 11]),
            horizon=1,
            pred_log_q10=np.full(2, -0.1),
            pred_log_q50=np.log1p(pred),
            pred_log_q90=np.full(2, 0.1),
            actual_log=np.log1p(actual),
            masks={"validation": np.array([True, True])},
        )["validation"]
        self.assertEqual(report["n_trades"], 0)
        self.assertAlmostEqual(report["cumulative_return"], 0.0)
        self.assertAlmostEqual(report["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
