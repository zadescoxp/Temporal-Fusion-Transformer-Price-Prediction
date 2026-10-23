# API

Run from this directory with the repository virtualenv active:

```bash
uvicorn app.main:app --reload --port 8000 --reload-dir app
```

Interactive docs: http://localhost:8000/docs

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/data/{ticker}?n=756` | Recent OHLCV and causal indicators |
| `POST` | `/api/predict` | Quantile paths for the requested horizons |
| `POST` | `/api/train` | Fit one ticker. Blocks until the run finishes |
| `GET` | `/api/interpretability/{ticker}` | Saved variable-selection weights, attention, and backtest |

The model is loaded on the first forecast and kept in memory. See the repository README for the split, the look-ahead rules, and the 15 bp per-side cost.
