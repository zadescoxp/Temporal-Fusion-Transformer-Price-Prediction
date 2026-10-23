"""Strict chronological cutoffs. No shuffling and no random splits."""

from __future__ import annotations

from datetime import date

import pandas as pd

IN_SAMPLE_END = pd.Timestamp("2023-12-31")
OOS_START = pd.Timestamp("2024-01-01")
# Last 20% of the in-sample time index is validation. The model is fit only
# on rows at or before the training cutoff, so validation targets never enter
# the loss or the normalizers.
VALIDATION_FRACTION = 0.20


def split_bounds(frame: pd.DataFrame) -> dict[str, int | date | None]:
    """Return integer time-index boundaries for train, validation, and OOS.

    `validation_start_idx` is the first time index whose targets belong to the
    held-out in-sample block. Training samples are built only from rows with
    `time_idx <= training_cutoff`.
    """

    if frame.empty:
        raise ValueError("Cannot split an empty frame.")
    dates = pd.to_datetime(frame["date"])
    time_idx = frame["time_idx"].astype(int)
    in_sample = dates <= IN_SAMPLE_END
    if int(in_sample.sum()) < 2:
        raise ValueError("In-sample window does not contain enough rows.")

    t_min = int(time_idx[in_sample].min())
    t_max = int(time_idx[in_sample].max())
    span = t_max - t_min
    validation_start_idx = t_min + int((1.0 - VALIDATION_FRACTION) * span)
    if validation_start_idx <= t_min or validation_start_idx > t_max:
        raise ValueError("Validation cutoff fell outside the in-sample window.")
    training_cutoff = validation_start_idx - 1

    oos = dates >= OOS_START
    oos_start_idx = int(time_idx[oos].min()) if bool(oos.any()) else None

    def _date_at(idx: int) -> date:
        match = frame.loc[time_idx == idx, "date"]
        return pd.Timestamp(match.iloc[0]).date()

    return {
        "training_cutoff": training_cutoff,
        "validation_start_idx": validation_start_idx,
        "in_sample_end_idx": t_max,
        "oos_start_idx": oos_start_idx,
        "training_cutoff_date": _date_at(training_cutoff),
        "validation_start_date": _date_at(validation_start_idx),
        "in_sample_end_date": _date_at(t_max),
        "oos_start_date": None if oos_start_idx is None else _date_at(oos_start_idx),
    }
