# Germany 40 Institutional Prediction Engine

A self-contained, mobile-friendly web app for forecasting the next directional
bias of **Germany 40 (DAX)** using a weighted multi-market macro model.

This is the **Python port** of the Pine Script indicator
`germany40_institutional_prediction_engine.pine` — same math, same buckets,
same forecast horizons, plus a real backtest engine.

## What's in the box

| Module | What it does |
|---|---|
| `app.py` | Streamlit UI (Live + Backtest tabs) |
| `engine/config.py` | Weights, periods, data-source selection |
| `engine/data.py` | `yfinance` (default) + `twelvedata` swap |
| `engine/indicators.py` | EMA / RSI / ROC (TradingView-equivalent) |
| `engine/scoring.py` | 7-market composite, 5-bucket classification, forecasts |
| `engine/backtest.py` | Long-flat backtest with equity curve + metrics |

## Run locally (5 min)

```bash
# 1. Clone / unzip into a folder
cd germany40_engine

# 2. Create a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser. To view it on
your phone on the same Wi-Fi, find your laptop's local IP (e.g. `192.168.1.42`)
and visit `http://192.168.1.42:8501` from your phone.

## Deploy to the public internet

### Option A — Streamlit Cloud (free, easiest)

1. Push the folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → pick the repo → entry point `app.py`.
4. Wait ~1 min.  You'll get a URL like
   `https://your-app.streamlit.app` that you can open from anywhere.

### Option B — Render / Railway / Fly.io

Each of these reads a `requirements.txt` automatically.  Start command:

```
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Option C — VPS (Hetzner / DigitalOcean)

```bash
# On the server
git clone <your repo>
cd germany40_engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nohup streamlit run app.py --server.port=80 --server.address=0.0.0.0 &
```

## Data source — yfinance vs Twelve Data

The app ships with **yfinance** (free, no key).  It covers all 7 markets
used in the model:

| Market | yfinance ticker | Twelve Data ticker |
|---|---|---|
| EUR/USD | `EURUSD=X` | `EUR/USD` |
| S&P 500 | `SPY` | `SPY` |
| Nasdaq 100 | `QQQ` | `QQQ` |
| Gold | `GC=F` | `XAU/USD` |
| DXY | `DX-Y.NYB` | `DXY` |
| VIX | `^VIX` | `VIX` |
| Germany 40 | `^GDAXI` | `DAX` |

**Switching to Twelve Data** is one dropdown + one API key in the sidebar:
1. Get a free key at [twelvedata.com](https://twelvedata.com)
   (800 requests/day, 8/min).
2. In the app sidebar → **Data Source → twelvedata** → paste key.

## Data interval — swing, intraday, scalping

A **Data interval** dropdown in the sidebar picks the bar size the engine
runs on. All 7 markets + the composite + all 3 forecast horizons are
recomputed on the selected interval. The math is the same — only the bar
size changes.

| Interval | Bars/day | Max lookback | Best for |
|---|---|---|---|
| `1d`  | 1 | 3 years | **Swing trading** (the spec default) |
| `1h`  | 24 | 60 days | **Intraday** (1-2 day holds) |
| `30m` | 32 | 30 days | Intraday / end-of-day |
| `15m` | 64 | 30 days | **Scalping / day trading** |
| `5m`  | 192 | 30 days | Scalping |
| `1m`  | 960 | 5 days | Ultra-short scalping |

**Important:** the 3 forecast horizons (Short / Medium / Long) are bar-count
horizons, not wall-clock horizons. So:

- On **1d** data: `Short = EMA10/20` ≈ 2-3 weeks. `Long = EMA50/200` ≈ 2-8 months.
- On **15m** data: `Short = EMA10/20` ≈ 2.5-5 hours. `Long = EMA50/200` ≈ 12-50 hours.

The header shows a tooltip explaining exactly what each horizon means in the
chosen interval.

The backtest holding-period input is also in **bars**, not days, so a
holding period of `5` on 1d data is 1 trading week, while `32` on 15m
data is one full trading day.

**yfinance limits** are respected automatically — switching to 15m caps the
lookback at 30 days because that's all yfinance returns for that interval.

**Twelve Data note:** their free tier is 8 requests/min.  7 markets × 1
intraday call each = 7 calls per refresh, so you'll hit the limit fast on
auto-refresh.  For serious intraday usage, upgrade to a paid tier or cache
heavily.

The code path swap is one line in `engine/data.py` (the `DataSourceFactory`).

## Backtest

The **🧪 Backtest** tab runs a long-flat strategy on historical daily data:

- **Long** DE40 when the model says `Bullish` or `Strong Bullish`
- **Cash** otherwise
- Default 5-day forward window (configurable 1–60 days)

You get:
- Strategy vs buy-and-hold equity curve
- CAGR, Sharpe, max drawdown, total return
- Per-signal hit rate and average forward return
- Score vs return scatter
- Last 30 signals table

Tweak weights in the sidebar → click **Rerun** in Streamlit → the whole
backtest recomputes in a few seconds.

## Phone / PWA notes

The app is mobile-responsive out of the box.  For a more "app-like" feel:

1. Open the deployed URL in your phone's browser.
2. **iOS Safari** → Share → *Add to Home Screen*.
3. **Android Chrome** → menu → *Add to Home Screen*.

It launches fullscreen, no browser chrome.

## Performance

- First data pull: 5–15 s (yfinance downloads 7 tickers).
- Cached for 5 min in Streamlit (`@st.cache_data(ttl=300)`).
- Composite + backtest: <1 s for 3 years of daily data.
- Suitable for free-tier hosting.

## Caveats

- yfinance is unofficial and can break when Yahoo changes its API.
  The Twelve Data swap is the recommended fallback.
- Backtest assumes zero commissions and zero slippage.  Use realistic
  numbers in your own broker before sizing real capital.
- Intraday intervals (15m/5m/1m) have limited lookback (5-30 days) on
  yfinance.  This is a yfinance limitation, not a bug.
- Twelve Data free tier (8 calls/min) is too tight for frequent
  intraday refreshes.  Upgrade to a paid tier or increase the cache TTL.
- The forecast horizons are bar-count horizons.  If you change the
  interval from 1d to 15m, the "Short" horizon stops being "15-30 min"
  and becomes "2.5-5 hours" — same EMA periods, different wall-clock
  duration.  Use the in-app tooltip to see exactly what each horizon
  means in your chosen interval.

## License

MIT — do whatever you want.
