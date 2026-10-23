"""The training bar fills with epochs and does not reach the end early."""

from __future__ import annotations

import unittest

from model.progress import training_fraction


class ProgressTests(unittest.TestCase):
    def test_first_batch_stays_near_the_start(self) -> None:
        fraction = training_fraction(1, 30, 1, 78)
        self.assertGreater(fraction, 0.08)
        self.assertLess(fraction, 0.12)

    def test_last_batch_of_the_plan_fills_the_training_slice(self) -> None:
        fraction = training_fraction(30, 30, 78, 78)
        self.assertAlmostEqual(fraction, 0.82)

    def test_unknown_batch_count_uses_the_epoch_only(self) -> None:
        early = training_fraction(1, 10, 0, None)
        late = training_fraction(10, 10, 0, None)
        self.assertAlmostEqual(early, 0.08)
        self.assertGreater(late, early)
        self.assertLess(late, 0.82)


if __name__ == "__main__":
    unittest.main()
