"""Chronological TimeSeriesDataSet splits.

Training rows end at the 80% time index of the in-sample window (through
2023-12-31). The validation dataset may read that earlier history as encoder
context, but its decoder targets start after the cutoff. Out-of-sample rows
are not passed into either constructor. Scalers live on the training dataset
and are reused, not refit, for validation and later walk-forward sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import StandardScaler

from model.config import (
    GROUP_ID,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    MIN_ENCODER_LENGTH,
    STATIC_CATEGORICALS,
    TARGET,
    TIME_IDX,
    TRAIN_END,
    UNKNOWN_REALS,
    VALIDATION_FRACTION,
)
from model.errors import DataError

TRAIN_END_TS = pd.Timestamp(TRAIN_END)


@dataclass
class SplitData:
    training: TimeSeriesDataSet
    validation: TimeSeriesDataSet
    training_cutoff: int
    insample_end: int
    oos_start: int | None
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    oos_start_date: str | None


def _fingerprint(dataset: TimeSeriesDataSet) -> str:
    """Numeric center/scale only. Pickle bytes differ across copies of the same fit."""
    norm = getattr(dataset.target_normalizer, "norm_", None)
    if not isinstance(norm, dict):
        return repr(norm)
    if "center" in norm and "scale" in norm and not hasattr(norm["center"], "to_numpy"):
        return f"{float(norm['center']):.12g}|{float(norm['scale']):.12g}"
    parts: list[str] = []
    for key in sorted(norm, key=str):
        value = norm[key]
        array = value.to_numpy() if hasattr(value, "to_numpy") else value
        parts.append(f"{key}:{np.array2string(np.asarray(array), precision=12)}")
    return "|".join(parts)


def make_dataset(
    frame: pd.DataFrame,
    *,
    training: TimeSeriesDataSet | None = None,
    min_prediction_idx: int | None = None,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
    min_prediction_length: int | None = None,
    predict: bool = False,
) -> TimeSeriesDataSet:
    """Build a dataset. Pass `training` to reuse fitted scalers and encoders."""
    data = frame.reset_index(drop=True)
    if data[TIME_IDX].duplicated().any():
        raise DataError("time_idx must be unique inside a series.")
    if not data[TIME_IDX].is_monotonic_increasing:
        raise DataError("time_idx must increase with the row order.")

    if min_prediction_length is None:
        min_prediction_length = max_prediction_length

    if training is None:
        # Every non-target real is z-scored with a StandardScaler fit on the
        # rows passed in here (the training cutoff only). Validation and
        # out-of-sample sets are built with from_dataset, which reuses those
        # scalers. The target uses its own GroupNormalizer, also fit on this slice.
        scalers = {
            name: StandardScaler()
            for name in (*KNOWN_REALS, *UNKNOWN_REALS)
            if name != TARGET
        }
        return TimeSeriesDataSet(
            data,
            time_idx=TIME_IDX,
            target=TARGET,
            group_ids=[GROUP_ID],
            min_encoder_length=MIN_ENCODER_LENGTH,
            max_encoder_length=MAX_ENCODER_LENGTH,
            min_prediction_length=min_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=list(STATIC_CATEGORICALS),
            time_varying_known_categoricals=list(KNOWN_CATEGORICALS),
            time_varying_known_reals=list(KNOWN_REALS),
            time_varying_unknown_reals=list(UNKNOWN_REALS),
            scalers=scalers,
            target_normalizer=GroupNormalizer(
                groups=[GROUP_ID],
                method="standard",
                center=True,
                scale_by_group=True,
                transformation=None,
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
            randomize_length=False,
            predict_mode=predict,
        )

    before = _fingerprint(training)
    created = TimeSeriesDataSet.from_dataset(
        training,
        data,
        stop_randomization=True,
        predict=predict,
        min_prediction_idx=min_prediction_idx,
        max_prediction_length=max_prediction_length,
        min_prediction_length=min_prediction_length,
    )
    after = _fingerprint(training)
    if before != after or _fingerprint(created) != before:
        raise DataError(
            "Target normalizer changed while building a non-training dataset. "
            "Validation or out-of-sample rows would have leaked into scaling."
        )
    return created


def build_splits(frame: pd.DataFrame) -> SplitData:
    ordered = frame.sort_values(TIME_IDX).reset_index(drop=True)
    insample = ordered[ordered["date"] <= TRAIN_END_TS].reset_index(drop=True)
    if insample.empty:
        raise DataError(f"No in-sample rows on or before {TRAIN_END}.")
    if ordered["date"].min() > pd.Timestamp("2012-01-01"):
        raise DataError("History starts too late for a 252-day encoder and a 252-day horizon.")

    cutoff = int(insample[TIME_IDX].quantile(1 - VALIDATION_FRACTION))
    insample_end = int(insample[TIME_IDX].max())
    earliest_cutoff = int(insample[TIME_IDX].min()) + MIN_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
    latest_cutoff = insample_end - MAX_PREDICTION_LENGTH
    if latest_cutoff <= earliest_cutoff:
        raise DataError("In-sample history is too short for the encoder and the 252-day horizon.")
    cutoff = min(max(cutoff, earliest_cutoff), latest_cutoff)

    training_rows = insample[insample[TIME_IDX] <= cutoff].reset_index(drop=True)
    if training_rows["date"].max() > TRAIN_END_TS:
        raise DataError("Training rows crossed the 2023-12-31 in-sample boundary.")

    training = make_dataset(training_rows)
    validation = make_dataset(
        insample,
        training=training,
        min_prediction_idx=cutoff + 1,
    )
    if len(training) == 0 or len(validation) == 0:
        raise DataError("The chronological split produced an empty training or validation set.")

    oos = ordered[ordered["date"] >= pd.Timestamp("2024-01-01")]
    oos_start = int(oos[TIME_IDX].min()) if len(oos) else None
    validation_start = insample.loc[insample[TIME_IDX] >= cutoff + 1, "date"].min()
    return SplitData(
        training=training,
        validation=validation,
        training_cutoff=cutoff,
        insample_end=insample_end,
        oos_start=oos_start,
        train_end_date=str(pd.Timestamp(training_rows["date"].max()).date()),
        validation_start_date=str(pd.Timestamp(validation_start).date()),
        validation_end_date=str(pd.Timestamp(insample["date"].max()).date()),
        oos_start_date=None if oos_start is None else str(pd.Timestamp(oos["date"].min()).date()),
    )
