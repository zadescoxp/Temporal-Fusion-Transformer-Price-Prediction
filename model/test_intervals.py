"""Band aggregation must not stack every extreme day."""

from __future__ import annotations

import unittest

import numpy as np

from model.intervals import cumulative_log_quantiles


class IntervalTests(unittest.TestCase):
    def test_one_step_band_is_the_daily_quantile(self) -> None:
        daily = np.array([[-0.02, 0.001, 0.03]])
        q10, q50, q90 = cumulative_log_quantiles(daily, 0, 1, 2)
        np.testing.assert_allclose(q10, [-0.02])
        np.testing.assert_allclose(q50, [0.001])
        np.testing.assert_allclose(q90, [0.03])

    def test_repeated_days_widen_with_sqrt_time(self) -> None:
        step = np.array([-0.02, 0.001, 0.03])
        daily = np.tile(step, (252, 1))
        q10, q50, q90 = cumulative_log_quantiles(daily, 0, 1, 2)
        scale = np.sqrt(252)
        self.assertAlmostEqual(q50[-1], 0.001 * 252)
        self.assertAlmostEqual(q50[-1] - q10[-1], 0.021 * scale)
        self.assertAlmostEqual(q90[-1] - q50[-1], 0.029 * scale)
        # Stacking the 0.9 path is the bug this guards against.
        self.assertLess(q90[-1], step[2] * 252 / 10)

    def test_batch_axis_is_preserved(self) -> None:
        daily = np.zeros((2, 4, 3))
        daily[..., 0] = -0.01
        daily[..., 2] = 0.01
        q10, q50, q90 = cumulative_log_quantiles(daily, 0, 1, 2)
        self.assertEqual(q50.shape, (2, 4))
        np.testing.assert_allclose(q50, 0)
        np.testing.assert_allclose(q10[:, -1], -0.01 * np.sqrt(4))
        np.testing.assert_allclose(q90[:, -1], 0.01 * np.sqrt(4))


if __name__ == "__main__":
    unittest.main()
