"""Historical OHLCV and the causal indicator columns the chart can inspect."""

from __future__ import annotations

from model.config import UNKNOWN_REALS, normalize_ticker
from model.features import prepare_history


def history_payload(ticker: str, n: int, refresh: bool = False) -> dict:
    ticker = normalize_ticker(ticker)
    bundle = prepare_history(ticker, refresh=refresh)
    tail = bundle.frame.tail(n)
    rows = []
    for record in tail.itertuples(index=False):
        row = {
            "date": record.date.strftime("%Y-%m-%d"),
            "open": float(record.open),
            "high": float(record.high),
            "low": float(record.low),
            "close": float(record.close),
            "volume": float(record.volume),
        }
        for name in UNKNOWN_REALS:
            row[name] = float(getattr(record, name))
        rows.append(row)
    return {"ticker": ticker, "rows": rows}
