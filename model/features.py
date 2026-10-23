"""Causal feature frame for one ticker.

Indicators use trailing windows only (pandas rolling, or `ta` EMA/Wilder
implementations with fillna disabled). Nothing is back-filled. Rows that
still contain warmup NaNs are dropped, so the first modeled session already
has a full lookback.

Price-level studies (moving averages, Ichimoku base line, ATR, MACD,
Awesome Oscillator, momentum, bull/bear power) are divided by the same-day
close. A dollar SMA fit on 2010 prices would not transfer to later prices.
Oscillators that already sit on a 0–100 or −100–0 scale are divided by 100.
The training dataset then z-scores every non-target input with a
StandardScaler fit on the training slice only.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import (
    AwesomeOscillatorIndicator,
    RSIIndicator,
    StochRSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.trend import (
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    IchimokuIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange

from model.calendar import is_holiday_adjacent, last_complete_session, next_trading_days
from model.config import (
    DATA_START,
    GROUP_ID,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET,
    TIME_IDX,
    UNKNOWN_REALS,
    normalize_ticker,
)
from model.errors import DataError

_YF_CACHE = RAW_DIR.parent / "yfinance-cache"
_YF_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(_YF_CACHE))

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, "HistoryBundle"]] = {}
_CACHE_TTL_SECONDS = 15 * 60

MODEL_COLUMNS = [TIME_IDX, GROUP_ID, TARGET, *KNOWN_CATEGORICALS, *KNOWN_REALS, *UNKNOWN_REALS]


@dataclass
class HistoryBundle:
    frame: pd.DataFrame
    earnings: set[date]
    dividends: set[date]


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    stamp = pd.Timestamp(value)
    if stamp is pd.NaT:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return stamp.date()


def _event_dates(values) -> set[date]:
    found: set[date] = set()
    for value in values:
        day = _as_date(value)
        if day is not None:
            found.add(day)
    return found


def fetch_events(ticker: str) -> tuple[set[date], set[date]]:
    """Event dates only. Reported EPS and dividend amounts are never read."""
    earnings: set[date] = set()
    dividends: set[date] = set()
    instrument = yf.Ticker(ticker)
    try:
        earnings_frame = instrument.get_earnings_dates(limit=100)
        if earnings_frame is not None and len(earnings_frame):
            earnings |= _event_dates(earnings_frame.index)
    except Exception as exc:
        logger.warning("Earnings calendar unavailable for %s: %s", ticker, exc)
    try:
        dividend_series = instrument.dividends
        if dividend_series is not None and len(dividend_series):
            dividends |= _event_dates(dividend_series.index)
    except Exception as exc:
        logger.warning("Dividend history unavailable for %s: %s", ticker, exc)
    try:
        calendar = instrument.calendar
        upcoming = None
        if isinstance(calendar, dict):
            upcoming = calendar.get("Ex-Dividend Date")
        elif isinstance(calendar, pd.DataFrame) and "Ex-Dividend Date" in calendar.index:
            upcoming = calendar.loc["Ex-Dividend Date"].iloc[0]
        day = _as_date(upcoming)
        if day is not None:
            dividends.add(day)
    except Exception as exc:
        logger.warning("Upcoming dividend calendar unavailable for %s: %s", ticker, exc)
    return earnings, dividends


def _flatten_download(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        chosen = None
        for level in range(frame.columns.nlevels):
            values = [str(item).lower() for item in frame.columns.get_level_values(level)]
            if "close" in values:
                chosen = level
                break
        if chosen is None:
            raise DataError("Price download did not include a close column.")
        frame.columns = [str(item).lower() for item in frame.columns.get_level_values(chosen)]
    else:
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame = frame.loc[:, ~pd.Index(frame.columns).duplicated()]
    frame = frame.reset_index()
    renamed = {column: column.lower().replace(" ", "_") for column in frame.columns}
    frame = frame.rename(columns=renamed)
    if "date" not in frame.columns:
        if "datetime" in frame.columns:
            frame = frame.rename(columns={"datetime": "date"})
        else:
            frame = frame.rename(columns={frame.columns[0]: "date"})
    if "adj_close" in frame.columns and "close" not in frame.columns:
        frame = frame.rename(columns={"adj_close": "close"})
    missing = [column for column in ("date", "open", "high", "low", "close", "volume") if column not in frame.columns]
    if missing:
        raise DataError(f"Price download is missing columns: {', '.join(missing)}.")
    return frame[["date", "open", "high", "low", "close", "volume"]].copy()


def download_ohlcv(ticker: str, start: str = DATA_START) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and len(raw):
                frame = _flatten_download(raw)
                break
            last_error = DataError(f"No prices returned for {ticker}.")
        except Exception as exc:
            last_error = exc
            logger.warning("Download attempt %s for %s failed: %s", attempt + 1, ticker, exc)
        time.sleep(1.2 * (attempt + 1))
    else:
        raise DataError(f"Could not download {ticker}: {last_error}") from last_error

    dates = pd.to_datetime(frame["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["date"] = dates.dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame[(frame["close"] > 0) & (frame["high"] >= frame["low"])]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    complete = pd.Timestamp(last_complete_session())
    frame = frame[frame["date"] <= complete]
    if frame.empty:
        raise DataError(f"No complete sessions available for {ticker}.")
    return frame.reset_index(drop=True)


def _calendar_frame(dates: pd.Series, earnings: set[date], dividends: set[date]) -> pd.DataFrame:
    days = dates.dt.date
    return pd.DataFrame(
        {
            "day_of_week": dates.dt.dayofweek.astype(str),
            "month": dates.dt.month.astype(str),
            "quarter": dates.dt.quarter.astype(str),
            "is_earnings_day": days.map(lambda day: 1.0 if day in earnings else 0.0).to_numpy(),
            "is_dividend_day": days.map(lambda day: 1.0 if day in dividends else 0.0).to_numpy(),
            "is_holiday": days.map(lambda day: 1.0 if is_holiday_adjacent(day) else 0.0).to_numpy(),
        },
        index=dates.index,
    )


MA_WINDOWS: tuple[int, ...] = (10, 20, 30, 50, 100, 200)
# TradingView MACD inputs are fast, slow, then signal. The request names the
# first two; the signal stays the usual 9-session EMA of the MACD line.
MACD_FAST = 10
MACD_SLOW = 12
MACD_SIGNAL = 9
# Elder bull/bear power. TradingView's built-in uses a 13-session EMA.
BULL_BEAR_WINDOW = 13
ICHIMOKU_CONVERSION = 9
ICHIMOKU_BASE = 26
ICHIMOKU_SPAN_B = 52


def _wma(series: pd.Series, window: int) -> pd.Series:
    """Linear weighted moving average. The newest sample has the largest weight."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if finite.any():
        start = int(np.argmax(finite))
        tail = values[start:]
        if len(tail) >= window and np.isfinite(tail).all():
            # np.convolve reverses the kernel, so pass the reversed weights.
            out[start + window - 1 :] = np.convolve(tail, weights[::-1], mode="valid")
            return pd.Series(out, index=series.index)
    if len(values) >= window:
        out = (
            series.rolling(window, min_periods=window)
            .apply(lambda window_values: float(np.dot(window_values, weights)), raw=True)
            .to_numpy(dtype=float)
        )
    return pd.Series(out, index=series.index)


def _hull(close: pd.Series, window: int = 9) -> pd.Series:
    """Hull moving average. Half-window and sqrt-window use half-up rounding."""
    half = max(int(np.floor(window / 2 + 0.5)), 1)
    sqrt_window = max(int(np.floor(np.sqrt(window) + 0.5)), 1)
    raw = 2.0 * _wma(close, half) - _wma(close, window)
    return _wma(raw, sqrt_window)


def _distance_from_close(indicator: pd.Series, close: pd.Series) -> pd.Series:
    return indicator / close - 1.0


def _over_close(indicator: pd.Series, close: pd.Series) -> pd.Series:
    return indicator / close


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Full stochastic %K and %D: 14-session range, then SMA 3, then SMA 3."""
    lowest = low.rolling(14, min_periods=14).min()
    highest = high.rolling(14, min_periods=14).max()
    raw = 100.0 * (close - lowest) / (highest - lowest)
    percent_k = raw.rolling(3, min_periods=3).mean()
    percent_d = percent_k.rolling(3, min_periods=3).mean()
    return percent_k / 100.0, percent_d / 100.0


def _vwma(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    volume_sum = volume.rolling(window, min_periods=window).sum()
    weighted = (close * volume).rolling(window, min_periods=window).sum()
    average = weighted / volume_sum.where(volume_sum > 0)
    return _distance_from_close(average, close)


def _add_indicators(frame: pd.DataFrame) -> None:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].clip(lower=0)

    for window in MA_WINDOWS:
        sma = SMAIndicator(close=close, window=window, fillna=False).sma_indicator()
        ema = EMAIndicator(close=close, window=window, fillna=False).ema_indicator()
        frame[f"sma_{window}"] = _distance_from_close(sma, close)
        frame[f"ema_{window}"] = _distance_from_close(ema, close)

    # visual=False keeps the base line on the session that produced it.
    # The displacement of 26 shifts the cloud, not the base line, and a
    # forward shift would read a value that has not been observed yet.
    ichimoku = IchimokuIndicator(
        high=high,
        low=low,
        window1=ICHIMOKU_CONVERSION,
        window2=ICHIMOKU_BASE,
        window3=ICHIMOKU_SPAN_B,
        visual=False,
        fillna=False,
    )
    frame["ichimoku_base"] = _distance_from_close(ichimoku.ichimoku_base_line(), close)
    frame["vwma_20"] = _vwma(close, volume, 20)
    frame["hull_9"] = _distance_from_close(_hull(close, 9), close)

    frame["rsi_14"] = RSIIndicator(close=close, window=14, fillna=False).rsi() / 100.0
    frame["stoch_k"], frame["stoch_d"] = _stochastic(high, low, close)
    frame["cci_20"] = (
        CCIIndicator(high=high, low=low, close=close, window=20, fillna=False).cci() / 100.0
    )
    frame["adx_14"] = (
        ADXIndicator(high=high, low=low, close=close, window=14, fillna=False).adx() / 100.0
    )
    frame["ao"] = _over_close(
        AwesomeOscillatorIndicator(high=high, low=low, window1=5, window2=34, fillna=False).awesome_oscillator(),
        close,
    )
    frame["momentum_10"] = _over_close(close - close.shift(10), close)
    macd = MACD(
        close=close,
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
        fillna=False,
    )
    frame["macd"] = _over_close(macd.macd(), close)
    frame["macd_signal"] = _over_close(macd.macd_signal(), close)
    stoch_rsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3, fillna=False)
    # ta already returns Stoch RSI on 0–1, which matches the other oscillators after /100.
    frame["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    frame["stoch_rsi_d"] = stoch_rsi.stochrsi_d()
    frame["williams_r"] = (
        WilliamsRIndicator(high=high, low=low, close=close, lbp=14, fillna=False).williams_r() / 100.0
    )
    bull_bear_ema = EMAIndicator(close=close, window=BULL_BEAR_WINDOW, fillna=False).ema_indicator()
    frame["bull_power"] = _over_close(high - bull_bear_ema, close)
    frame["bear_power"] = _over_close(low - bull_bear_ema, close)
    frame["ultimate_osc"] = (
        UltimateOscillator(
            high=high,
            low=low,
            close=close,
            window1=7,
            window2=14,
            window3=28,
            fillna=False,
        ).ultimate_oscillator()
        / 100.0
    )
    frame["atr_14"] = _over_close(
        AverageTrueRange(high=high, low=low, close=close, window=14, fillna=False).average_true_range(),
        close,
    )


def build_model_frame(prices: pd.DataFrame, ticker: str, earnings: set[date], dividends: set[date]) -> pd.DataFrame:
    frame = prices.copy()
    close = frame["close"]
    # Trailing only: shift looks backward, rolling windows end at the current row.
    frame[TARGET] = np.log(close / close.shift(1))
    frame["log_volume"] = np.log1p(frame["volume"].clip(lower=0))
    _add_indicators(frame)
    for window, name in ((5, "vol_5"), (20, "vol_20"), (60, "vol_60")):
        frame[name] = frame[TARGET].rolling(window, min_periods=window).std(ddof=1)
    frame["ret_lag_1"] = frame[TARGET].shift(1)
    frame["ret_lag_5"] = frame[TARGET].shift(5)
    frame["ret_lag_21"] = frame[TARGET].shift(21)
    calendar = _calendar_frame(frame["date"], earnings, dividends)
    frame = pd.concat([frame, calendar], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    required = [TARGET, *UNKNOWN_REALS, *KNOWN_REALS, *KNOWN_CATEGORICALS]
    # Unique while preserving order. TARGET is already inside UNKNOWN_REALS.
    required = list(dict.fromkeys(required))
    frame = frame.dropna(subset=required).reset_index(drop=True)
    if len(frame) < MAX_HISTORY_FLOOR:
        raise DataError(
            f"{ticker} has {len(frame)} usable rows after indicator warmup. "
            "The encoder needs several years of daily history."
        )
    validate_feature_frame(frame)
    frame[TIME_IDX] = np.arange(len(frame), dtype=np.int64)
    frame[GROUP_ID] = ticker
    for column in KNOWN_CATEGORICALS:
        frame[column] = frame[column].astype(str)
    for column in (*KNOWN_REALS, *UNKNOWN_REALS):
        frame[column] = frame[column].astype(float)
    return frame


MAX_HISTORY_FLOOR = 504


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Fail the run if a lag or a rolling window can see the future."""
    log_return = frame[TARGET].to_numpy()
    if not np.allclose(frame["ret_lag_1"].to_numpy()[1:], log_return[:-1]):
        raise DataError("ret_lag_1 is not the previous log return.")
    if not np.allclose(frame["ret_lag_5"].to_numpy()[5:], log_return[:-5]):
        raise DataError("ret_lag_5 is not a causal shift.")
    if not np.allclose(frame["ret_lag_21"].to_numpy()[21:], log_return[:-21]):
        raise DataError("ret_lag_21 is not a causal shift.")
    expected = float(frame[TARGET].iloc[-5:].std(ddof=1))
    actual = float(frame["vol_5"].iloc[-1])
    if not np.isclose(expected, actual, rtol=1e-6, atol=1e-8):
        raise DataError("vol_5 is not the trailing five-session standard deviation.")
    columns = [*KNOWN_REALS, *UNKNOWN_REALS]
    if frame[columns].isna().any().any():
        raise DataError("Feature frame still contains NaNs after warmup was dropped.")
    # A forgotten close-relative scale leaves dollar prices here. A 200-day
    # average sits near the price; the stored feature is the gap, near zero.
    if float(frame["sma_200"].abs().median()) > 5:
        raise DataError("sma_200 is not scaled relative to the close.")
    if float(frame["rsi_14"].abs().median()) > 5:
        raise DataError("rsi_14 was not divided by 100.")
    if not frame["williams_r"].between(-1.05, 0.05).all():
        raise DataError("williams_r is outside the normalized −1 to 0 range.")
    if not frame["stoch_k"].between(-0.05, 1.05).all():
        raise DataError("stoch_k is outside the normalized 0 to 1 range.")


def prepare_history(ticker: str, refresh: bool = False) -> HistoryBundle:
    ticker = normalize_ticker(ticker)
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(ticker)
        if cached and not refresh and now - cached[0] < _CACHE_TTL_SECONDS:
            bundle = cached[1]
            return HistoryBundle(bundle.frame.copy(), set(bundle.earnings), set(bundle.dividends))

    prices = download_ohlcv(ticker)
    earnings, dividends = fetch_events(ticker)
    frame = build_model_frame(prices, ticker, earnings, dividends)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(RAW_DIR / f"{ticker.lower()}.parquet", index=False)
    frame.to_parquet(PROCESSED_DIR / f"{ticker.lower()}.parquet", index=False)
    bundle = HistoryBundle(frame, earnings, dividends)
    with _CACHE_LOCK:
        _CACHE[ticker] = (time.time(), bundle)
        return HistoryBundle(bundle.frame.copy(), set(bundle.earnings), set(bundle.dividends))


def append_future(bundle: HistoryBundle, periods: int) -> pd.DataFrame:
    """Add the decoder rows the TFT needs at inference.

    Known calendar fields are filled for every future session. Unknown
    features are zeros. The training StandardScaler is already fit, so those
    zeros do not move historical values, and the decoder variable-selection
    network is not fed the unknown columns.
    """
    history = bundle.frame
    last = history.iloc[-1]
    last_day = pd.Timestamp(last["date"]).date()
    last_idx = int(last[TIME_IDX])
    group = str(last[GROUP_ID])
    rows: list[dict[str, object]] = []
    for step, day in enumerate(next_trading_days(last_day, periods), start=1):
        stamp = pd.Timestamp(day)
        row: dict[str, object] = {
            "date": stamp,
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": np.nan,
            TIME_IDX: last_idx + step,
            GROUP_ID: group,
            "day_of_week": str(stamp.dayofweek),
            "month": str(stamp.month),
            "quarter": str(stamp.quarter),
            "is_earnings_day": 1.0 if day in bundle.earnings else 0.0,
            "is_dividend_day": 1.0 if day in bundle.dividends else 0.0,
            "is_holiday": 1.0 if is_holiday_adjacent(day) else 0.0,
        }
        for column in UNKNOWN_REALS:
            row[column] = 0.0
        rows.append(row)
    extended = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    for column in KNOWN_CATEGORICALS:
        extended[column] = extended[column].astype(str)
    extended[GROUP_ID] = extended[GROUP_ID].astype(str)
    for column in (*KNOWN_REALS, *UNKNOWN_REALS):
        extended[column] = extended[column].astype(float)
    extended[TIME_IDX] = extended[TIME_IDX].astype(np.int64)
    return extended
