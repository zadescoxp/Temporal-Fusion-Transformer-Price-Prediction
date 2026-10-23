"""Shared horizons, costs, feature contract, and on-disk layout.

The feature lists are the contract between training and inference. Changing a
name here requires a retrain; the forecast service reads the same lists.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BACKEND_DIR / "models"

DATA_START = "2010-01-01"
TRAIN_END = "2023-12-31"
OOS_START = "2024-01-01"
VALIDATION_FRACTION = 0.20

HORIZONS: tuple[int, ...] = (1, 5, 21, 63, 126, 252)
HORIZON_LABELS: dict[int, str] = {
    1: "1d",
    5: "1w",
    21: "1m",
    63: "3m",
    126: "6m",
    252: "1y",
}

MAX_ENCODER_LENGTH = 252
MIN_ENCODER_LENGTH = 126
MAX_PREDICTION_LENGTH = 252
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# 10 bps brokerage + 5 bps slippage, charged on each side.
COMMISSION_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
COST_PER_SIDE = COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE
ROUND_TRIP_COST = 2 * COST_PER_SIDE

HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16
LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32

TARGET = "log_return"
GROUP_ID = "group_id"
TIME_IDX = "time_idx"

# Calendar fields the decoder is allowed to see. Binaries stay numeric so a
# 0/1 value that was rare in training cannot crash the categorical encoder.
KNOWN_CATEGORICALS: tuple[str, ...] = ("day_of_week", "month", "quarter")
KNOWN_REALS: tuple[str, ...] = ("is_earnings_day", "is_dividend_day", "is_holiday")

# Observed only up to the forecast origin. The target is included, which is
# what the TFT encoder reads as the history of the series.
#
# Price-level studies are stored as a distance from that session's close.
# Oscillators that live on a 0–100 (or −100–0) scale are divided by 100.
# The training dataset then applies a StandardScaler fit on the training
# slice only. The target stays a raw log return; GroupNormalizer scales it.
UNKNOWN_REALS: tuple[str, ...] = (
    "log_return",
    "log_volume",
    "sma_10",
    "sma_20",
    "sma_30",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_10",
    "ema_20",
    "ema_30",
    "ema_50",
    "ema_100",
    "ema_200",
    "ichimoku_base",
    "vwma_20",
    "hull_9",
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "cci_20",
    "adx_14",
    "ao",
    "momentum_10",
    "macd",
    "macd_signal",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "williams_r",
    "bull_power",
    "bear_power",
    "ultimate_osc",
    "atr_14",
    "vol_5",
    "vol_20",
    "vol_60",
    "ret_lag_1",
    "ret_lag_5",
    "ret_lag_21",
)

STATIC_CATEGORICALS: tuple[str, ...] = (GROUP_ID,)

TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


def normalize_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not TICKER_RE.match(value):
        raise ValueError(
            "Ticker must be 1–10 characters: letters, digits, dots, or hyphens."
        )
    return value


def artifact_dir(ticker: str) -> Path:
    path = MODELS_DIR / normalize_ticker(ticker).lower()
    path.mkdir(parents=True, exist_ok=True)
    return path
