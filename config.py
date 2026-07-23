"""
Configuration container for the DE40 engine.

Mirrors the user inputs from the Pine Script.  The defaults match the
institutional model in the spec, but every value can be overridden from
the Streamlit sidebar (see app.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict


# ----------------------------------------------------------------------------- 
# Default constants - tune these to taste
# -----------------------------------------------------------------------------
DEFAULT_DATA_SOURCE = "yfinance"   # "yfinance" or "twelvedata"
DEFAULT_TWELVE_DATA_KEY = ""       # populate via env var or sidebar

DEFAULT_WEIGHTS: Dict[str, float] = {
    "eurusd": 25.0,
    "sp500":  20.0,
    "nasdaq": 15.0,
    "gold":   10.0,
    "dxy":    10.0,
    "vix":    10.0,
    "de40":   10.0,
}

DEFAULT_PERIODS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_len":  14,
    "roc_len":  20,
}

DEFAULT_FORECASTS = {
    "short":  {"ema_fast": 10, "ema_slow": 20, "rsi": 7},
    "medium": {"ema_fast": 20, "ema_slow": 50, "rsi": 14},
    "long":   {"ema_fast": 50, "ema_slow": 200, "rsi": 21},
}

MARKET_LABELS = {
    "eurusd": "EUR/USD",
    "sp500":  "S&P 500",
    "nasdaq": "Nasdaq 100",
    "gold":   "Gold",
    "dxy":    "DXY",
    "vix":    "VIX",
    "de40":   "Germany 40",
}

# Markets that are NEGATIVELY correlated with DE40 in the model.
# A "bullish" reading on these should *subtract* from the composite.
NEGATIVE_CORRELATIONS = {"eurusd", "gold", "dxy", "vix"}


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class Config:
    data_source: str = DEFAULT_DATA_SOURCE
    twelvedata_api_key: str = DEFAULT_TWELVE_DATA_KEY
    interval: str = "1d"   # 1m, 5m, 15m, 30m, 1h, 1d
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    periods: Dict[str, int]  = field(default_factory=lambda: dict(DEFAULT_PERIODS))
    forecasts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        k: dict(v) for k, v in DEFAULT_FORECASTS.items()
    })

    # --- helpers -------------------------------------------------------------
    def weight_sum(self) -> float:
        return sum(max(0.0, v) for v in self.weights.values()) or 1.0

    def normalized_weights(self) -> Dict[str, float]:
        s = self.weight_sum()
        return {k: max(0.0, v) / s for k, v in self.weights.items()}

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_sidebar(cls) -> "Config":
        """
        Build a Config from the Streamlit sidebar.  Imported lazily so that
        the engine can be used headless (e.g. from the backtest CLI) without
        Streamlit being installed.
        """
        import streamlit as st
        cfg = cls()

        with st.sidebar:
            st.subheader("Data Source")
            src = st.selectbox(
                "Provider",
                ["yfinance", "twelvedata"],
                index=0 if cfg.data_source == "yfinance" else 1,
                help=(
                    "yfinance is free, no key, covers all 7 markets.  "
                    "Twelve Data needs a free API key from twelvedata.com."
                ),
            )
            cfg.data_source = src
            if src == "twelvedata":
                cfg.twelvedata_api_key = st.text_input(
                    "Twelve Data API key", value=cfg.twelvedata_api_key, type="password"
                )

            # --- Data interval (intraday / daily toggle) --------------------
            # Lazy import to avoid circular import at module load time.
            # Works in both package and flat layouts.
            try:
                from .data import INTERVAL_CONFIG
            except ImportError:
                from data import INTERVAL_CONFIG
            interval_opts = list(INTERVAL_CONFIG.keys())
            current_idx = interval_opts.index(cfg.interval) if cfg.interval in interval_opts else 0
            cfg.interval = st.selectbox(
                "Data interval (bar size)",
                interval_opts,
                index=current_idx,
                help=(
                    "1d = swing trading (years of history).  "
                    "1h = intraday (60-day lookback).  "
                    "15m/5m/1m = scalping (7-30 day lookback).  "
                    "All 3 forecast horizons are computed on this interval; "
                    "they represent bar-count horizons, not wall-clock time."
                ),
                key="interval_select",
            )
            cfg._interval_lookback = INTERVAL_CONFIG[cfg.interval]["lookback_days"]

            st.subheader("Weights (% — auto-normalized)")
            new_w = {}
            for k, label in MARKET_LABELS.items():
                new_w[k] = st.slider(
                    label, 0, 100, int(cfg.weights[k]), step=1, key=f"w_{k}"
                )
            cfg.weights = new_w

            with st.expander("Indicator periods", expanded=False):
                cfg.periods["ema_fast"] = st.number_input(
                    "EMA fast", 2, 200, cfg.periods["ema_fast"])
                cfg.periods["ema_slow"] = st.number_input(
                    "EMA slow", 5, 400, cfg.periods["ema_slow"])
                cfg.periods["rsi_len"]  = st.number_input(
                    "RSI length", 2, 100, cfg.periods["rsi_len"])
                cfg.periods["roc_len"]  = st.number_input(
                    "ROC length", 1, 200, cfg.periods["roc_len"])

            with st.expander("Forecast horizons", expanded=False):
                st.caption("Short (15-30m)")
                cfg.forecasts["short"]["ema_fast"]  = st.number_input("S EMA fast", 2, 50, cfg.forecasts["short"]["ema_fast"])
                cfg.forecasts["short"]["ema_slow"]  = st.number_input("S EMA slow", 5, 100, cfg.forecasts["short"]["ema_slow"])
                cfg.forecasts["short"]["rsi"]       = st.number_input("S RSI",      2, 50,  cfg.forecasts["short"]["rsi"])
                st.caption("Medium (1-4h)")
                cfg.forecasts["medium"]["ema_fast"] = st.number_input("M EMA fast", 2, 100, cfg.forecasts["medium"]["ema_fast"])
                cfg.forecasts["medium"]["ema_slow"] = st.number_input("M EMA slow", 5, 200, cfg.forecasts["medium"]["ema_slow"])
                cfg.forecasts["medium"]["rsi"]      = st.number_input("M RSI",      2, 50,  cfg.forecasts["medium"]["rsi"])
                st.caption("Long (1-3d)")
                cfg.forecasts["long"]["ema_fast"]   = st.number_input("L EMA fast", 5, 200, cfg.forecasts["long"]["ema_fast"])
                cfg.forecasts["long"]["ema_slow"]   = st.number_input("L EMA slow", 10, 400, cfg.forecasts["long"]["ema_slow"])
                cfg.forecasts["long"]["rsi"]        = st.number_input("L RSI",      2, 100, cfg.forecasts["long"]["rsi"])

            if st.button("Reset to defaults"):
                st.rerun()

        return cfg
