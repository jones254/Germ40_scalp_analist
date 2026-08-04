"""
Streamlit entry point for the Germany 40 Institutional Prediction Engine.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    push the repo (with all .py files at the root) to GitHub and connect
    the repo to share.streamlit.io.  Entry point: `app.py`.

This version uses FLAT IMPORTS so it works whether you uploaded the engine
modules as `engine/*.py` (proper package) OR as `*.py` at the repo root
(flat layout, which is what GitHub's web uploader produces).  It tries the
flat layout first, then falls back to the `engine.` package.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Make sure the script directory is on sys.path so flat-import deployments
# (no `engine/` folder) work out of the box.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Robust imports — try flat layout first (e.g. config.py at the repo root),
# fall back to the `engine.` package layout (e.g. engine/config.py).
# ---------------------------------------------------------------------------
def _try_import():
    """Return a dict of names that work for either deployment layout."""
    try:
        from config import Config, MARKET_LABELS  # noqa: F401
        from data import DataSourceFactory       # noqa: F401
        from scoring import (                    # noqa: F401
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from backtest import BacktestConfig, run_backtest  # noqa: F401
        from trades import run_trade_engine, TradeConfig as EngineTradeConfig  # noqa: F401
        from trade_ui import render_trade_dashboard        # noqa: F401
        return {
            "Config": Config,
            "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score,
            "forecasts": forecasts,
            "forecast_classify": forecast_classify,
            "market_regime": market_regime,
            "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig,
            "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "flat",
        }
    except ImportError:
        from engine.config import Config, MARKET_LABELS  # noqa: F401
        from engine.data import DataSourceFactory       # noqa: F401
        from engine.scoring import (                    # noqa: F401
            composite_score, forecasts, forecast_classify,
            market_regime, flow_meter,
        )
        from engine.backtest import BacktestConfig, run_backtest  # noqa: F401
        from engine.trades import run_trade_engine, TradeConfig as EngineTradeConfig  # noqa: F401
        from engine.trade_ui import render_trade_dashboard        # noqa: F401
        return {
            "Config": Config,
            "MARKET_LABELS": MARKET_LABELS,
            "DataSourceFactory": DataSourceFactory,
            "composite_score": composite_score,
            "forecasts": forecasts,
            "forecast_classify": forecast_classify,
            "market_regime": market_regime,
            "flow_meter": flow_meter,
            "BacktestConfig": BacktestConfig,
            "run_backtest": run_backtest,
            "run_trade_engine": run_trade_engine,
            "EngineTradeConfig": EngineTradeConfig,
            "render_trade_dashboard": render_trade_dashboard,
            "layout": "engine",
        }


_IMPORTS = _try_import()
Config                  = _IMPORTS["Config"]
MARKET_LABELS           = _IMPORTS["MARKET_LABELS"]
DataSourceFactory       = _IMPORTS["DataSourceFactory"]
composite_score         = _IMPORTS["composite_score"]
forecasts               = _IMPORTS["forecasts"]
forecast_classify       = _IMPORTS["forecast_classify"]
market_regime           = _IMPORTS["market_regime"]
flow_meter              = _IMPORTS["flow_meter"]
BacktestConfig          = _IMPORTS["BacktestConfig"]
run_backtest            = _IMPORTS["run_backtest"]
run_trade_engine        = _IMPORTS["run_trade_engine"]
EngineTradeConfig       = _IMPORTS["EngineTradeConfig"]
render_trade_dashboard  = _IMPORTS["render_trade_dashboard"]
_LAYOUT                 = _IMPORTS["layout"]


# -----------------------------------------------------------------------------
# Page config & theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DE40 Institutional Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Soft custom CSS to make the dashboard feel a bit more "Bloomberg-lite"
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .stMetric > div { padding: 0.5rem 0.75rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
        }
        .pill { display: inline-block; padding: 2px 10px; border-radius: 12px;
                font-size: 0.78rem; font-weight: 600; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
SIGNAL_COLOURS = {
    "Strong Bullish":  "#006400",
    "Bullish":         "#2E8B57",
    "Neutral":         "#DAA520",
    "Bearish":         "#B22222",
    "Strong Bearish":  "#8B0000",
}

REGIME_COLOURS = {
    "Risk-On":     "#2E8B57",
    "Risk-Off":    "#B22222",
    "Transition":  "#DAA520",
}


@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def _fetch(source_name: str, api_key: str, interval: str, lookback_days: int):
    cfg = Config(data_source=source_name, twelvedata_api_key=api_key, interval=interval)
    src = DataSourceFactory.create(cfg)
    return src.fetch_all(lookback_days=lookback_days, interval=interval)


def _score_alignment_check(data):
    """Sanity check: warn the user if some markets are missing."""
    missing = [m for m in MARKET_LABELS if m not in data or len(data[m]) == 0]
    return missing


def _gauge(value, title, min_val=0, max_val=100, suffix=""):
    """Half-circle gauge for the flow meter / composite."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 28}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar":  {"color": "#1f77b4"},
            "steps": [
                {"range": [min_val, 40],         "color": "#f8d7da"},
                {"range": [40,         60],      "color": "#fff3cd"},
                {"range": [60,         max_val], "color": "#d4edda"},
            ],
        },
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=0))
    return fig


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
config = Config.from_sidebar()


# -----------------------------------------------------------------------------
# Pull data
# -----------------------------------------------------------------------------
# Robust import — works in both flat layout (`data.py` at the root) and
# `engine.data` package layout.
try:
    from engine.data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG
except ImportError:
    from data import interval_lookback_days, interval_bar_label, INTERVAL_CONFIG

INTERVAL_LABELS = {
    "1d":  "Daily (1d) — swing trading",
    "1h":  "Hourly (1h) — intraday",
    "30m": "30-min — intraday",
    "15m": "15-min — intraday / scalping",
    "5m":  "5-min — scalping",
    "1m":  "1-min — scalping (limited history)",
}

with st.spinner(f"Loading {config.interval} market data…"):
    try:
        lb = interval_lookback_days(config.interval)
        data = _fetch(config.data_source, config.twelvedata_api_key, config.interval, lb)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

missing = _score_alignment_check(data)
if missing:
    st.warning(f"Missing data for: {', '.join(missing)}. "
               "Try switching the data source in the sidebar.")
    if "de40" not in data or len(data["de40"]) == 0:
        st.error("DE40 (Germany 40) is required.  Cannot continue.")
        st.stop()


# -----------------------------------------------------------------------------
# Compute composite + forecasts + regime + flow
# -----------------------------------------------------------------------------
with st.spinner("Computing composite score…"):
    score_result = composite_score(data, config)
    fcasts = forecasts(data, config)
    f_labels = {}
    f_conf   = {}
    for name, s in fcasts.items():
        lbl, conf = forecast_classify(s)
        f_labels[name] = lbl
        f_conf[name]   = conf
    regime = market_regime(score_result.per_market)
    flow   = flow_meter(
        composite   = score_result.composite,
        vix_close   = data["vix"]["Close"]   if "vix"   in data else pd.Series(15.0, index=score_result.composite.index),
        dxy_close   = data["dxy"]["Close"]   if "dxy"   in data else pd.Series(100.0, index=score_result.composite.index),
        nasdaq_score= score_result.per_market["nasdaq"] if "nasdaq" in score_result.per_market else pd.Series(0.0, index=score_result.composite.index),
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("📈 Germany 40 Institutional Prediction Engine")
st.caption(
    f"Interval: **{INTERVAL_LABELS.get(config.interval, config.interval)}** · "
    f"Data: **{config.data_source}** · "
    f"Last bar: **{data['de40'].index[-1].strftime('%Y-%m-%d %H:%M')}** · "
    f"Markets loaded: **{len(data)}/7**"
)

# Helpful hint: what each forecast horizon means in the current timeframe.
_interval_meta = INTERVAL_CONFIG[config.interval]
_bpd = _interval_meta["bars_per_day"]
_blabel = _interval_meta["bar_label"]
def _humanize(bars: int) -> str:
    """Turn a bar-count into a rough human duration for the current interval."""
    if config.interval == "1d":
        return f"~{bars} days"
    if config.interval == "1h":
        return f"~{bars/24:.1f} days" if bars >= 24 else f"~{bars} hours"
    hours = bars * {"30m": 0.5, "15m": 0.25, "5m": 5/60, "1m": 1/60}.get(config.interval, 1.0)
    if hours < 1:
        return f"~{int(hours*60)} min"
    if hours < 24:
        return f"~{hours:.1f} hours"
    return f"~{hours/24:.1f} days"

_s_ef = config.forecasts["short"]["ema_fast"]; _s_es = config.forecasts["short"]["ema_slow"]
_m_ef = config.forecasts["medium"]["ema_fast"]; _m_es = config.forecasts["medium"]["ema_slow"]
_l_ef = config.forecasts["long"]["ema_fast"]; _l_es = config.forecasts["long"]["ema_slow"]

with st.expander("ℹ️  What do the forecast horizons mean in this timeframe?", expanded=False):
    st.markdown(
        f"""
All three horizons are computed on **{config.interval} bars** (the same
interval you selected in the sidebar).  The numbers below are bar counts
of the EMA fast/slow pair, translated into rough wall-clock durations:

| Horizon | EMA pair | Approx. duration in **{config.interval}** bars |
|---|---|---|
| **Short**  | EMA{_s_ef}/{_s_es} + RSI{config.forecasts["short"]["rsi"]} | {_humanize(_s_ef)} → {_humanize(_s_es)} |
| **Medium** | EMA{_m_ef}/{_m_es} + RSI{config.forecasts["medium"]["rsi"]} | {_humanize(_m_ef)} → {_humanize(_m_es)} |
| **Long**   | EMA{_l_ef}/{_l_es} + RSI{config.forecasts["long"]["rsi"]} | {_humanize(_l_ef)} → {_humanize(_l_es)} |

The **composite score** uses the same 7-market, 40/35/25 blend on
{config.interval} bars.
        """
    )


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_live, tab_backtest, tab_trades, tab_about = st.tabs(
    ["📊  Live", "🧪  Backtest", "🎯  Trades", "ℹ️  About"]
)


# =============================================================================
# LIVE TAB
# =============================================================================
with tab_live:
    latest = score_result.composite.index[-1]
    last_score   = float(score_result.composite.loc[latest])
    last_label   = str(score_result.label.loc[latest])
    last_conf    = float(score_result.confidence.loc[latest])
    last_color   = SIGNAL_COLOURS[last_label]

    # ----- Big score + signal -----------------------------------------------
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        st.markdown(
            f"""
            <div style="background:{last_color}22;border-left:6px solid {last_color};
                        padding:14px 18px;border-radius:8px">
              <div style="font-size:0.9rem;color:#666">COMPOSITE SIGNAL</div>
              <div style="font-size:1.6rem;font-weight:700;color:{last_color}">
                {last_label}
              </div>
              <div style="font-size:0.85rem;color:#444">as of {latest.strftime('%Y-%m-%d')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.metric("Composite Score",  f"{last_score:+.1f}")
    with col_c:
        st.metric("Confidence",       f"{last_conf:.1f}%")
    with col_d:
        st.metric(
            "Institutional Flow",
            f"{float(flow.loc[latest]):.0f}/100",
            delta=f"{float(flow.loc[latest]) - 50:+.0f} vs neutral",
        )

    st.markdown("")

    # ----- Per-market breakdown table ---------------------------------------
    st.subheader("Market breakdown")
    contribs = score_result.contributions
    weights  = config.normalized_weights()
    rows = []
    for mkt, label in MARKET_LABELS.items():
        if mkt not in score_result.per_market:
            continue
        s = float(score_result.per_market[mkt].loc[latest])
        w = weights[mkt] * 100
        c = float(contribs[mkt].loc[latest])
        direction = "▼" if c < 0 else "▲"
        rows.append({
            "Market":      label,
            "Asset Score": f"{s:+.1f}",
            "Weight":      f"{w:.1f}%",
            "Contribution": f"{c:+.1f}",
            "Direction":   direction,
        })
    df_mkts = pd.DataFrame(rows).set_index("Market")

    def _color_contrib(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 20:   return "background-color: #2E8B5730"
        if v < -20:  return "background-color: #B2222230"
        return "background-color: #DAA52030"

    st.dataframe(
        df_mkts.style.map(_color_contrib, subset=["Contribution"]),
        use_container_width=True, height=290,
    )

    # ----- Live entry / SL / TP setup ---------------------------------------
    # Mirror the trade-engine math for the *current* bar, no look-ahead.
    # This is read-only "what would the engine do right now" output.
    try:
        from indicators import atr as _atr  # flat layout
    except ImportError:
        try:
            from engine.indicators import atr as _atr
        except ImportError:
            _atr = None

    if _atr is not None and "de40" in data and len(data["de40"]) > 30:
        de40_now = data["de40"]
        atr_series = _atr(de40_now["High"], de40_now["Low"], de40_now["Close"], 14)
        price_now = float(de40_now["Close"].iloc[-1])
        atr_now   = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

        # Defaults must match the trade engine defaults; we recompute
        # entry/SL/TP for whatever the live composite is suggesting.
        _k_sl, _k_tp, _pb_frac = 1.5, 3.0, 0.5

        if atr_now > 0 and last_label in ("Strong Bullish", "Bullish", "Bearish", "Strong Bearish"):
            side = 1 if "Bullish" in last_label else -1
            entry_limit = price_now - side * _pb_frac * atr_now
            stop_dist   = _k_sl * atr_now
            target_dist = _k_tp * atr_now
            stop_price  = entry_limit - side * stop_dist
            target_price= entry_limit + side * target_dist
            rr          = target_dist / stop_dist

            # Position size (1% risk, default equity)
            equity      = 10_000.0
            risk_dollars= equity * 0.01
            size        = risk_dollars / stop_dist

            side_label  = "LONG 🟢" if side == 1 else "SHORT 🔴"
            stop_color  = "#B22222"
            tgt_color   = "#006400"

            st.subheader("🎯 Live entry setup (current bar)")
            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.metric("Action",  side_label)
            ec2.metric("Entry (limit)", f"{entry_limit:,.2f}",
                       delta=f"-{_pb_frac*atr_now:.1f} from close")
            ec3.metric("Stop",    f"{stop_price:,.2f}",
                       delta=f"{stop_dist:.1f} pts", delta_color="inverse")
            ec4.metric("Target",  f"{target_price:,.2f}",
                       delta=f"+{target_dist:.1f} pts")
            ec5.metric("Size (1% risk)", f"{size:.3f}",
                       help="Contracts/lots at 1% account risk on a $10k account. "
                            f"Risk = ${risk_dollars:.0f} per trade.")

            # Visual on price
            st.caption(
                f"Entry at **{entry_limit:,.2f}** · "
                f"Stop **{stop_price:,.2f}** ({stop_dist:.1f} pts) · "
                f"Target **{target_price:,.2f}** ({target_dist:.1f} pts) · "
                f"R:R = **{rr:.2f}**"
            )

            # Price chart with overlays (last 60 bars)
            lookback = min(60, len(de40_now))
            chart_df = de40_now.tail(lookback)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"], high=chart_df["High"],
                low=chart_df["Low"],   close=chart_df["Close"],
                name="DE40", increasing_line_color="#2E8B57",
                decreasing_line_color="#B22222",
            ))
            # Horizontal lines for entry / SL / TP
            for level, color, name in [
                (entry_limit,  "#1f77b4", f"Entry {entry_limit:,.0f}"),
                (stop_price,   stop_color, f"Stop {stop_price:,.0f}"),
                (target_price, tgt_color,  f"Target {target_price:,.0f}"),
            ]:
                fig.add_hline(y=level, line_color=color, line_width=2,
                              line_dash="dash", annotation_text=name,
                              annotation_position="right",
                              annotation_font_color=color)
            fig.update_layout(
                height=380, template="plotly_white",
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="DE40", xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"ℹ️ No entry setup on current bar — composite is **{last_label}** "
                    f"(score {last_score:+.1f}). Trade engine only fires on "
                    f"Bullish / Bearish buckets. ATR={atr_now:.1f} pts.")
    else:
        st.info("Live entry setup unavailable (ATR not yet warmed up or imports failed).")

    st.markdown("")

    # ----- Forecasts + regime -----------------------------------------------
    st.subheader("Forecasts")
    f1, f2, f3 = st.columns(3)
    for col, name, lbl in [(f1, "short", "Short (15-30m)"),
                           (f2, "medium", "Medium (1-4h)"),
                           (f3, "long",   "Long (1-3d)")]:
        with col:
            v = f_labels[name].loc[latest]
            c = f_conf[name].loc[latest]
            color = SIGNAL_COLOURS[v]
            st.markdown(
                f"""
                <div style="background:{color}1a;border-left:5px solid {color};
                            padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">{lbl}</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{color}">{v}</div>
                  <div style="font-size:0.78rem;color:#444">Confidence: {c:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- Regime + flow ---------------------------------------------------
    st.subheader("Regime & flow")
    r1, r2 = st.columns([1, 1])
    with r1:
        reg_now = regime.loc[latest]
        st.markdown(
            f"""<div style="background:{REGIME_COLOURS[reg_now]}1a;border-left:5px solid {REGIME_COLOURS[reg_now]};
                        padding:10px 12px;border-radius:6px">
                  <div style="font-size:0.78rem;color:#666">Market Regime</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{REGIME_COLOURS[reg_now]}">{reg_now}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "Risk-On: SP>20 ∧ VIX<−10 ∧ Gold<−10 · "
            "Risk-Off: SP<−20 ∧ VIX>10 ∧ Gold>10 · otherwise Transition"
        )
    with r2:
        st.plotly_chart(
            _gauge(float(flow.loc[latest]), "Institutional Flow Meter"),
            use_container_width=True,
        )

    # ----- Historical composite chart --------------------------------------
    st.subheader("Composite history")
    window = st.slider("Lookback (days)", 30, 730, 180, key="live_lb")
    hist = score_result.composite.tail(window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Composite",
                             line=dict(color="#1f77b4", width=2)))
    for level, color, name in [
        ( 70, "#006400", "Strong Bull"),
        ( 40, "#2E8B57", "Bull"),
        (-40, "#B22222", "Bear"),
        (-70, "#8B0000", "Strong Bear"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.5,
                      annotation_text=name, annotation_position="right")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Composite Score", xaxis_title=None,
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Mini price charts ------------------------------------------------
    st.subheader("Underlying price action")
    chart_cols = st.columns(2)
    for i, (mkt, label) in enumerate(MARKET_LABELS.items()):
        if mkt not in data:
            continue
        close = data[mkt]["Close"].tail(120)
        with chart_cols[i % 2]:
            fig = go.Figure(go.Scatter(x=close.index, y=close.values, name=label,
                                       line=dict(color="#1f77b4", width=1.5)))
            fig.update_layout(height=160, margin=dict(l=10, r=10, t=20, b=0),
                              title=label, template="plotly_white",
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_backtest:
    st.subheader("🧪 Historical backtest")
    st.caption(
        f"Strategy: long DE40 when the model says **Bullish / Strong Bullish**, "
        f"cash otherwise. Returns are computed on **{config.interval} bars**. "
        f"No look-ahead — the signal at bar T only uses bars up to T.  "
        f"⚠️ Intraday backtests use limited history "
        f"({interval_lookback_days(config.interval)} days max)."
    )

    # ----- Backtest controls ------------------------------------------------
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start = st.date_input("Start",  value=datetime(2018, 1, 1),
                              min_value=datetime(2000, 1, 1),
                              max_value=datetime.now())
    with bc2:
        end   = st.date_input("End",    value=datetime.now(),
                              min_value=datetime(2000, 1, 1),
                              max_value=datetime.now() + timedelta(days=1))
    with bc3:
        # Holding period is in BARS (since with intraday data, "5 days" is meaningless)
        _max_hold = {"1d": 250, "1h": 500, "30m": 500, "15m": 500, "5m": 1000, "1m": 1000}.get(config.interval, 500)
        _default_hold = {"1d": 5, "1h": 12, "30m": 24, "15m": 32, "5m": 96, "1m": 240}.get(config.interval, 5)
        hold  = st.number_input(
            f"Holding period ({_blabel}s)",
            1, _max_hold, _default_hold,
            help=(
                "Forward return window measured in bars.  "
                "5 daily bars = 1 trading week.  "
                "12 hourly bars = 1.5 trading days.  "
                "32 15-min bars = 1 trading day."
            ),
        )

    bt_cfg = BacktestConfig(
        start=start.isoformat(),
        end=end.isoformat(),
        holding_period=int(hold),
    )

    with st.spinner("Running backtest…"):
        try:
            result = run_backtest(data, config, bt_cfg)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    # ----- Headline metrics -------------------------------------------------
    st.markdown("##### Performance vs buy & hold")
    m = result.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Strategy CAGR",       f"{m['Strategy CAGR %']:.2f}%",
               delta=f"{m['Strategy CAGR %'] - m['Benchmark CAGR %']:.2f}% vs BH")
    mc2.metric("Strategy Sharpe",      f"{m['Strategy Sharpe']:.2f}",
               delta=f"{m['Strategy Sharpe'] - m['Benchmark Sharpe']:.2f}")
    mc3.metric("Strategy Max DD",      f"{m['Strategy Max DD %']:.2f}%",
               delta=f"{m['Strategy Max DD %'] - m['Benchmark Max DD %']:.2f}% vs BH",
               delta_color="inverse")
    mc4.metric("Strategy Total Ret",   f"{m['Strategy Total Ret %']:.2f}%",
               delta=f"{m['Strategy Total Ret %'] - m['Benchmark Total Ret %']:.2f}% vs BH")

    # ----- Equity curve -----------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Strategy (long-flat)", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                             name="Buy & Hold DE40",      line=dict(color="#888", width=1.5, dash="dot")))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=0),
                      yaxis_title="Equity ($)", xaxis_title=None,
                      template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Per-bucket stats -------------------------------------------------
    st.markdown("##### Per-signal statistics (forward return, %)")
    st.dataframe(result.summary, use_container_width=True)

    # ----- Recent signals ---------------------------------------------------
    st.markdown("##### Last 30 signals")
    st.dataframe(
        result.signals.tail(30).iloc[::-1]
            .assign(composite=lambda d: d["composite"].round(1),
                    fwd_return=lambda d: (d["fwd_return"] * 100).round(2))
            .rename(columns={"fwd_return": "fwd_ret_%"}),
        use_container_width=True, height=380,
    )

    # ----- Score distribution ----------------------------------------------
    st.markdown("##### Score distribution & forward return")
    fig = px.scatter(
        result.signals.dropna(subset=["fwd_return"]).reset_index(),
        x="composite", y="fwd_return",
        color="signal",
        color_discrete_map=SIGNAL_COLOURS,
        labels={"composite": "Composite Score", "fwd_return": "Forward Return"},
        opacity=0.6,
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TRADES TAB  — R-based execution engine
# =============================================================================
with tab_trades:
    st.subheader("🎯 Trade Engine — R-based execution")
    st.caption(
        "Pullback entries · ATR stops · structural stop blend · automatic "
        "breakeven at +1R · risk-based sizing · both directions."
    )

    # ---- Trade engine controls --------------------------------------------
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        risk_pct = st.number_input(
            "Risk per trade (% equity)",
            0.1, 5.0, 1.0, step=0.1,
            help="1% is the institutional default. Higher = more volatile PnL.",
            key="trade_risk",
        ) / 100.0
    with tc2:
        k_sl = st.number_input(
            "Stop (× ATR)", 0.5, 5.0, 1.5, step=0.1, key="trade_k_sl",
            help="Stop distance in ATR multiples from the entry. "
                 "Tighter = smaller wins/losses, more frequent stops.",
        )
    with tc3:
        k_tp = st.number_input(
            "Target (× ATR)", 1.0, 10.0, 3.0, step=0.1, key="trade_k_tp",
            help="Take-profit distance in ATR multiples from the entry.",
        )
    with tc4:
        min_rr = st.number_input(
            "Min R:R filter", 1.0, 5.0, 1.5, step=0.1, key="trade_min_rr",
            help="Signals with R:R below this are skipped.",
        )

    tc5, tc6, tc7, tc8 = st.columns(4)
    with tc5:
        use_be = st.checkbox(
            "Breakeven at +1R", value=True, key="trade_be",
            help="When price reaches 1R of profit, move stop to entry. "
                 "Losses that briefly went +1R become 0R instead of -1R.",
        )
    with tc6:
        swing_lb = st.number_input(
            "Swing lookback (bars)", 0, 50, 10, step=1, key="trade_swing",
            help="Window for structural stop (recent swing low/high). "
                 "0 = disable structural stop, ATR only.",
        )
    with tc7:
        pb_frac = st.number_input(
            "Pullback depth (× ATR)", 0.0, 2.0, 0.5, step=0.1, key="trade_pb",
            help="How deep the limit entry sits from the signal close. "
                 "0 = market entry on next bar; 0.5 = half-ATR pullback.",
        )
    with tc8:
        entry_win = st.number_input(
            "Entry window (bars)", 1, 20, 3, step=1, key="trade_ew",
            help="How long the pullback limit order stays working before "
                 "expiring unfilled.",
        )

    trade_cfg = EngineTradeConfig(
        risk_per_trade=float(risk_pct),
        k_sl=float(k_sl),
        k_tp=float(k_tp),
        min_rr=float(min_rr),
        use_breakeven_be=bool(use_be),
        swing_lookback=int(swing_lb),
        pullback_atr_frac=float(pb_frac),
        entry_window=int(entry_win),
    )

    with st.spinner("Running trade engine…"):
        try:
            trade_log = run_trade_engine(
                de40=data["de40"],
                composite=score_result.composite,
                labels=score_result.label,
                confidence=score_result.confidence,
                cfg=trade_cfg,
                initial_equity=10_000.0,
            )
            render_trade_dashboard(trade_log, data["de40"], config)
        except Exception as e:
            st.error(f"Trade engine failed: {e}")
            st.exception(e)


# =============================================================================
# ABOUT TAB
# =============================================================================
with tab_about:
    st.markdown(
        """
        ### What this is
        A multi-market, weighted macro engine for forecasting the next
        directional bias of **Germany 40 (DAX)**.

        The math is the **Python port** of the Pine Script indicator
        `germany40_institutional_prediction_engine.pine` — same 40/35/25
        blend, same 5-bucket classification, same forecast horizons.

        ### Data sources
        - **yfinance** (default) — free, no key, covers all 7 markets.
        - **Twelve Data** — faster, needs a free API key
          (twelvedata.com).  Swap with one dropdown in the sidebar.

        ### Deploy
        1. Push the folder to GitHub.
        2. Go to **share.streamlit.io** and connect the repo.
        3. Add `app.py` as the entry point.
        4. Open the URL on your phone — works great in mobile browsers.
        For PWA / "Add to Home Screen" support, see README.md.

        ### Limits
        - Free yfinance tier is delayed 15 min and occasionally rate-limited.
        - Twelve Data free tier = 800 requests/day, 8/min.
        - Backtest assumes daily bars, no commissions, no slippage.
        """
    )

    st.markdown("#### Engine modules")
    st.code(
        """
        engine/
        ├── __init__.py
        ├── config.py     # weights, periods, data source
        ├── data.py       # yfinance + Twelve Data abstraction
        ├── indicators.py # EMA, RSI, ROC (Pine-equivalent)
        ├── scoring.py    # 7-market composite + forecasts
        ├── backtest.py   # long-flat backtest engine
        ├── trades.py     # ★ R-based trade engine (entries, SL, TP, BE)
        └── trade_ui.py   # ★ Trade dashboard (metrics, equity, log)
        """,
        language="text",
    )
