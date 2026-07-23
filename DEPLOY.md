# Flat-layout deployment

Upload every file in this folder to the ROOT of your GitHub repo
(overwriting your existing `app.py`, `config.py`, `data.py`, etc.).

The `app.py` in this zip auto-detects whether you have a flat layout
or a proper `engine/` package layout, so it works either way.

## What was fixed in this build
- Added try/except to the `INTERVAL_CONFIG` import in `config.py` so it
  works in both flat and package layouts.  This fixes the
  `ImportError: attempted relative import with no known parent package`
  error you saw on Streamlit Cloud.
- All other engine modules already had this fix from the previous build.

## Files
- `app.py`              - Streamlit entry point
- `config.py`           - Weights, periods, data-source, **interval**
- `data.py`             - YFinance + Twelve Data (interval-aware)
- `indicators.py`       - EMA / RSI / ROC
- `scoring.py`          - 7-market composite engine
- `backtest.py`         - Long-flat backtest
- `requirements.txt`    - Python dependencies
- `README.md`           - Full docs
- `DEPLOY.md`           - This file

## Steps
1. In your GitHub repo, **replace** these 7 files with the ones in this folder:
   `app.py`, `config.py`, `data.py`, `scoring.py`, `backtest.py`,
   `indicators.py`, `requirements.txt`.
2. The old `__init__.py` and root `config.toml` are harmless but unused —
   you can leave or delete them.
3. Streamlit Cloud will auto-rebuild on push.  If it doesn't, click
   **Reboot** in the app dashboard.
