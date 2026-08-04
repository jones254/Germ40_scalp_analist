"""
Trade execution engine for the DE40 Institutional Prediction Engine.

Takes the composite signal series from `scoring.composite_score` and
the DE40 OHLC dataframe, then produces a complete trade log with
entries, stops, targets, exits, and R-multiple based performance.

Design
------
* Interval-agnostic: ATR period and swing lookback are expressed in
  bars; the same code works on 1m through 1d.
* Pullback entries: we don't enter at the signal bar's close, we
  place a limit at 0.5*ATR in the direction of the trend and let
  it fill (or expire after `entry_window` bars).
* Risk-based sizing: position size = risk_budget / stop_distance,
  so a tight stop = bigger size, wide stop = smaller size.
* Both directions: long and short are symmetric.
* No look-ahead: every decision at bar T uses only data up to T.

Outputs
-------
A `TradeLog` dataclass carrying:
  - trades:        per-trade detail (DataFrame)
  - equity_curve:  mark-to-market equity path
  - metrics:       dict of headline R-based stats
  - signals_used:  signal series that was actually traded on
                   (after pullback entry logic, may differ from raw)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .indicators import atr
except ImportError:
    from indicators import atr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class TradeConfig:
    """
    All knobs you need to tune the entry / SL / TP logic.

    Defaults are tuned to be robust across 1m-1d intervals; the only
    things most users ever change are `risk_per_trade` and the
    R:R target.
    """
    # Risk per trade as a fraction of equity (e.g. 0.01 = 1%).
    risk_per_trade: float = 0.01

    # Stop loss = k_sl * ATR(atr_len) measured from the entry price.
    k_sl: float = 1.5

    # Take profit = k_tp * ATR(atr_len) measured from the entry price.
    k_tp: float = 3.0

    # ATR period in bars.  14 is the standard across all intervals.
    atr_len: int = 14

    # Pullback depth as a fraction of ATR (0.5 = half an ATR pullback).
    pullback_atr_frac: float = 0.5

    # How many bars the pullback limit order stays working before
    # it expires and we cancel the entry.
    entry_window: int = 3

    # Swing-lookback window in bars for the structural stop blend.
    # Set to 0 to disable structural stops and use pure ATR.
    swing_lookback: int = 10

    # Whether to move stop to breakeven after 1R of profit.
    use_breakeven_be: bool = True

    # Which composite-score buckets to trade.
    # Each entry: bucket -> dict with side + size_scale.
    # size_scale multiplies the risk budget, so half conviction
    # trades at 0.5x risk.
    bucket_plan: Dict[str, Dict] = field(default_factory=lambda: {
        "Strong Bullish":  {"side":  1, "size_scale": 1.0},
        "Bullish":         {"side":  1, "size_scale": 0.5},
        "Neutral":         {"side":  0, "size_scale": 0.0},
        "Bearish":         {"side": -1, "size_scale": 0.5},
        "Strong Bearish":  {"side": -1, "size_scale": 1.0},
    })

    # Min R:R we accept.  Signals whose stop+target geometry makes
    # R:R < this threshold are skipped (would-be losers get filtered).
    min_rr: float = 1.5

    # Max stop as a fraction of price (sanity guard against
    # illiquid-bar blowups, e.g. 0.03 = 3% of price).
    max_stop_pct: float = 0.03


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
@dataclass
class TradeLog:
    trades: pd.DataFrame           # one row per closed trade
    equity_curve: pd.Series        # mark-to-market equity path
    metrics: Dict[str, float]      # R-based headline stats
    signal_used: pd.Series         # the bucketed signal that drove entries
    skipped: pd.DataFrame          # signals we filtered (low R:R, no fill, etc.)


# -----------------------------------------------------------------------------
# Helper: find the most recent swing low / high in a window
# -----------------------------------------------------------------------------
def _swing_low(low: pd.Series, lookback: int) -> pd.Series:
    """Rolling min over the last `lookback` bars (exclusive of current)."""
    return low.shift(1).rolling(lookback, min_periods=1).min()


def _swing_high(high: pd.Series, lookback: int) -> pd.Series:
    """Rolling max over the last `lookback` bars (exclusive of current)."""
    return high.shift(1).rolling(lookback, min_periods=1).max()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def run_trade_engine(
    de40: pd.DataFrame,
    composite: pd.Series,
    labels: pd.Series,
    confidence: pd.Series,
    cfg: TradeConfig = TradeConfig(),
    initial_equity: float = 10_000.0,
) -> TradeLog:
    """
    Walk the DE40 bars, fire entries on bucket signals, manage stops
    and targets, and produce a closed-trade log.

    Parameters
    ----------
    de40 : DataFrame
        OHLCV with at least columns High, Low, Close.
    composite : Series
        Composite score aligned to de40.index.
    labels : Series
        5-bucket classification (Strong Bullish ... Strong Bearish).
    confidence : Series
        Per-bar confidence in percent.
    cfg : TradeConfig
        Tuning knobs (see above).
    initial_equity : float
        Starting account size in account currency.

    Returns
    -------
    TradeLog
    """
    # ---- 1. Prep indicators -------------------------------------------------
    df = de40.copy()
    df["atr"]      = atr(df["High"], df["Low"], df["Close"], cfg.atr_len)
    df["swing_lo"] = _swing_low(df["Low"],  cfg.swing_lookback) if cfg.swing_lookback > 0 else np.nan
    df["swing_hi"] = _swing_high(df["High"], cfg.swing_lookback) if cfg.swing_lookback > 0 else np.nan

    # Build a `bucket_plan` lookup that's NaN-safe
    def _plan(lbl):
        if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
            return {"side": 0, "size_scale": 0.0}
        return cfg.bucket_plan.get(lbl, {"side": 0, "size_scale": 0.0})

    # ---- 2. Walk forward, manage open position -----------------------------
    equity       = initial_equity
    equity_path  = []
    open_pos: Optional[dict] = None
    pending: Optional[dict] = None  # unfilled pullback limit order
    trades: list  = []
    skipped: list = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        equity_path.append((ts, equity))

        # ------------------------------------------------------------------
        # A) If a position is open, check exits first.
        # ------------------------------------------------------------------
        if open_pos is not None:
            hi = float(row["High"])
            lo = float(row["Low"])

            # Breakeven stop adjustment after 1R of profit (using H/L)
            if cfg.use_breakeven_be and not open_pos["be_moved"]:
                if   open_pos["side"] ==  1 and lo <= open_pos["entry"]:
                    pass  # bar touched entry from above but didn't go further
                if   open_pos["side"] ==  1 and hi >= open_pos["entry"] + open_pos["r_distance"]:
                    open_pos["stop"] = open_pos["entry"]
                    open_pos["be_moved"] = True
                elif open_pos["side"] == -1 and hi >= open_pos["entry"]:
                    pass
                elif open_pos["side"] == -1 and lo <= open_pos["entry"] - open_pos["r_distance"]:
                    open_pos["stop"] = open_pos["entry"]
                    open_pos["be_moved"] = True

            exit_price = None
            exit_reason = None

            if open_pos["side"] == 1:
                # Check stop first (conservative intrabar ordering: stop before target)
                if lo <= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                    exit_reason = "stop"
                elif hi >= open_pos["target"]:
                    exit_price = open_pos["target"]
                    exit_reason = "target"
            else:  # short
                if hi >= open_pos["stop"]:
                    exit_price = open_pos["stop"]
                    exit_reason = "stop"
                elif lo <= open_pos["target"]:
                    exit_price = open_pos["target"]
                    exit_reason = "target"

            # If still open, mark to market at close
            if exit_price is not None:
                pnl = (exit_price - open_pos["entry"]) * open_pos["side"] * open_pos["size"]
                equity += pnl
                r_pnl  = pnl / open_pos["risk_dollars"] if open_pos["risk_dollars"] > 0 else 0.0
                trades.append({
                    "entry_time":  open_pos["entry_time"],
                    "exit_time":   ts,
                    "side":        "long" if open_pos["side"] == 1 else "short",
                    "entry":       round(open_pos["entry"], 2),
                    "stop":        round(open_pos["stop"], 2),
                    "target":      round(open_pos["target"], 2),
                    "exit":        round(exit_price, 2),
                    "size":        round(open_pos["size"], 4),
                    "risk_dollars":round(open_pos["risk_dollars"], 2),
                    "pnl":         round(pnl, 2),
                    "r_multiple":  round(r_pnl, 3),
                    "bars_held":   open_pos["bars_held"],
                    "exit_reason": exit_reason,
                    "signal":      open_pos["signal"],
                    "confidence":  open_pos["confidence"],
                    "be_moved":    open_pos["be_moved"],
                })
                open_pos = None
            else:
                # Position still open, increment hold counter, mark to market
                open_pos["bars_held"] += 1
                mtm = (price - open_pos["entry"]) * open_pos["side"] * open_pos["size"]
                # Don't add unrealized PnL to equity_path (we record realized
                # equity at each bar; alternative would be to add MTM).
                # Keeping it simple: equity_path tracks realized equity,
                # open positions are visible in the trades log.
                _ = mtm  # noqa: F841 (kept for future MTM-equity mode)

        # ------------------------------------------------------------------
        # B) If a pending pullback order is alive, check for fill.
        # ------------------------------------------------------------------
        if open_pos is None and pending is not None:
            lo = float(row["Low"])
            hi = float(row["High"])
            filled = False

            if pending["side"] == 1 and lo <= pending["entry_limit"]:
                filled_price = pending["entry_limit"]  # assume fill at limit
                filled = True
            elif pending["side"] == -1 and hi >= pending["entry_limit"]:
                filled_price = pending["entry_limit"]
                filled = True

            if filled:
                stop_dist = pending["stop_distance"]
                tgt_dist  = pending["target_distance"]
                # Sanity check: stop must not exceed max_stop_pct of price
                if stop_dist / price > cfg.max_stop_pct:
                    skipped.append({"time": ts, "reason": "stop_too_wide",
                                    "signal": pending["signal"]})
                    pending = None
                else:
                    risk_dollars = equity * cfg.risk_per_trade * pending["size_scale"]
                    size = risk_dollars / stop_dist
                    open_pos = {
                        "side":         pending["side"],
                        "entry_time":   ts,
                        "entry":        filled_price,
                        "stop":         filled_price - pending["side"] * stop_dist,
                        "target":       filled_price + pending["side"] * tgt_dist,
                        "size":         size,
                        "risk_dollars": risk_dollars,
                        "r_distance":   stop_dist,
                        "bars_held":    0,
                        "be_moved":     False,
                        "signal":       pending["signal"],
                        "confidence":   pending["confidence"],
                    }
                    pending = None
            else:
                pending["bars_pending"] += 1
                if pending["bars_pending"] >= cfg.entry_window:
                    skipped.append({"time": ts, "reason": "entry_expired",
                                    "signal": pending["signal"]})
                    pending = None

        # ------------------------------------------------------------------
        # C) If flat and no pending order, look for a NEW signal on this bar.
        # ------------------------------------------------------------------
        if open_pos is None and pending is None:
            if ts not in labels.index:
                continue
            lbl = labels.loc[ts]
            plan = _plan(lbl)
            if plan["side"] == 0:
                continue

            atr_v = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
            if atr_v <= 0:
                skipped.append({"time": ts, "reason": "no_atr", "signal": lbl})
                continue

            # Distances
            stop_dist   = cfg.k_sl * atr_v
            target_dist = cfg.k_tp * atr_v

            # Structural stop blend: use the TIGHTER of ATR / swing stop.
            # A real trader would never widen a stop just because a swing is
            # far away — they only use the swing if it gives a closer invalidation.
            if cfg.swing_lookback > 0:
                if plan["side"] == 1 and not pd.isna(row["swing_lo"]):
                    struct_stop = max(0.0, price - row["swing_lo"])
                    stop_dist = min(stop_dist, struct_stop)
                if plan["side"] == -1 and not pd.isna(row["swing_hi"]):
                    struct_stop = max(0.0, row["swing_hi"] - price)
                    stop_dist = min(stop_dist, struct_stop)

            rr = target_dist / stop_dist if stop_dist > 0 else 0.0
            if rr < cfg.min_rr:
                skipped.append({"time": ts, "reason": f"rr_{rr:.2f}_below_min",
                                "signal": lbl})
                continue

            # Pullback entry limit
            entry_limit = price - plan["side"] * cfg.pullback_atr_frac * atr_v

            pending = {
                "side":            plan["side"],
                "entry_limit":     entry_limit,
                "stop_distance":   stop_dist,
                "target_distance": target_dist,
                "size_scale":      plan["size_scale"],
                "bars_pending":    0,
                "signal":          lbl,
                "confidence":      float(confidence.loc[ts]) if ts in confidence.index else np.nan,
            }

    # -------------------------------------------------------------------------
    # Build outputs
    # -------------------------------------------------------------------------
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.set_index("exit_time").sort_index()

    eq = pd.Series([e for _, e in equity_path],
                   index=[t for t, _ in equity_path],
                   name="equity")

    metrics = _compute_r_metrics(trades_df, initial_equity, eq)
    skipped_df = pd.DataFrame(skipped).set_index("time") if skipped else pd.DataFrame()

    return TradeLog(
        trades=trades_df,
        equity_curve=eq,
        metrics=metrics,
        signal_used=labels,  # raw labels; the engine filtered internally
        skipped=skipped_df,
    )


# -----------------------------------------------------------------------------
# R-based metrics
# -----------------------------------------------------------------------------
def _compute_r_metrics(
    trades: pd.DataFrame,
    initial_equity: float,
    equity: pd.Series,
) -> Dict[str, float]:
    """Headline performance metrics expressed in R-multiples where possible."""
    if trades.empty:
        return {
            "n_trades": 0, "hit_rate_pct": 0.0, "expectancy_r": 0.0,
            "avg_win_r": 0.0, "avg_loss_r": 0.0, "profit_factor": 0.0,
            "max_consec_wins": 0, "max_consec_losses": 0,
            "total_pnl": 0.0, "final_equity": float(initial_equity),
            "return_pct": 0.0,
        }

    r = trades["r_multiple"]
    wins  = r[r > 0]
    losses = r[r < 0]

    # Max consecutive wins / losses
    signs = np.sign(r.values)
    max_w = max_l = cur_w = cur_l = 0
    for s in signs:
        if s > 0:
            cur_w += 1; cur_l = 0
        elif s < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss   = -losses.sum() if len(losses) else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    final_eq = float(equity.iloc[-1]) if len(equity) else initial_equity
    total_pnl = float(trades["pnl"].sum())

    return {
        "n_trades":         int(len(trades)),
        "hit_rate_pct":     round(float((r > 0).mean() * 100), 2),
        "expectancy_r":     round(float(r.mean()), 3),
        "avg_win_r":        round(float(wins.mean()) if len(wins) else 0.0, 3),
        "avg_loss_r":       round(float(losses.mean()) if len(losses) else 0.0, 3),
        "profit_factor":    round(pf, 2) if np.isfinite(pf) else 99.99,
        "max_consec_wins":  int(max_w),
        "max_consec_losses":int(max_l),
        "total_pnl":        round(total_pnl, 2),
        "final_equity":     round(final_eq, 2),
        "return_pct":       round((final_eq / initial_equity - 1) * 100, 2),
    }


# -----------------------------------------------------------------------------
# Convenience: pretty-print the trade log
# -----------------------------------------------------------------------------
def trade_log_summary(trade_log: TradeLog, last_n: int = 20) -> pd.DataFrame:
    """Return the last N closed trades formatted for display."""
    if trade_log.trades.empty:
        return pd.DataFrame()
    cols = ["entry_time", "side", "signal", "entry", "stop", "target",
            "exit", "size", "r_multiple", "pnl", "exit_reason", "bars_held"]
    available = [c for c in cols if c in trade_log.trades.columns]
    return trade_log.trades[available].tail(last_n).iloc[::-1]
