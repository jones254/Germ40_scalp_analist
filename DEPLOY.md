# Flat-layout deployment

Upload every file in this folder to the ROOT of your GitHub repo
(overwriting your existing `app.py`, `config.py`, `data.py`, etc.).

The `app.py` in this zip auto-detects whether you have a flat layout
or a proper `engine/` package layout, so it works either way.

## What changed vs the previous zip
- New **Data interval** dropdown in the sidebar (1d / 1h / 30m / 15m / 5m / 1m).
- All 7 markets + composite + 3 forecast horizons now run on the
  selected interval (was daily-only before).
- yfinance lookback limits are respected per interval.
- Backtest holding period is now in **bars** (was days).
- Header shows what each horizon means in the current timeframe.

## Files
- `app.py`              - Streamlit entry point
- `config.py`           - Weights & data-source config (with interval field)
- `data.py`             - YFinance + Twelve Data fetchers (interval-aware)
- `indicators.py`       - EMA / RSI / ROC
- `scoring.py`          - 7-market composite engine
- `backtest.py`         - Long-flat backtest
- `requirements.txt`    - Python dependencies
- `README.md`           - Full docs
- `DEPLOY.md`           - This file

## Steps
1. In your GitHub repo, delete the old `__init__.py` and `config.toml`
   at the root (unused / ignored).
2. Replace all .py files with the ones in this folder.
3. On share.streamlit.io the entry point is `app.py`.
4. Pick your interval in the sidebar (start with `1h` for intraday).
