"""
Composite scoring engine.

This is the Python port of the Pine Script logic in
`germany40_institutional_prediction_engine.pine`.

The high-level pipeline for a fresh batch of market data:

    1. Compute Trend / Momentum / Strength for each market
    2. Blend with the 40/35/25 weights -> asset score
    3. Invert the sign of negatively-correlated assets
    4. Weighted sum, normalised -> composite in [-100, +100]
    5. Classify into 5 buckets (Strong Bull -> Strong Bear)
    6. Compute three forecast horizons on DE40
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Works in both `engine.scoring` package layout AND flat file layout
try:
    from .indicators import ema, rsi, roc
    from .config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS
except ImportError:
    from indicators import ema, rsi, roc
    from config import Config, MARKET_LABELS, NEGATIVE_CORRELATIONS


# -----------------------------------------------------------------------------
# Per-market asset score
# -----------------------------------------------------------------------------
def asset_score(close: pd.Series, periods: Dict[str, int]) -> pd.Series:
    """
    40% Trend + 35% Momentum + 25% Strength, clamped to [-100, +100].
    """
    t = (close.ewm(span=periods["ema_fast"], adjust=False, min_periods=periods["ema_fast"]).mean()
         > close.ewm(span=periods["ema_slow"], adjust=False, min_periods=periods["ema_slow"]).mean())
    e1 = close.ewm(span=periods["ema_fast"], adjust=False, min_periods=periods["ema_fast"]).mean()
    e2 = close.ewm(span=periods["ema_slow"], adjust=False, min_periods=periods["ema_slow"]).mean()
    t_score = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))

    r = rsi(close, periods["rsi_len"])
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_score = np.where(r > 60, pos, np.where(r < 40, neg, soft))

    rc = roc(close, periods["roc_len"])
    s_score = (rc * 20.0).clip(-100.0, 100.0)

    raw = t_score * 0.40 + m_score * 0.35 + s_score * 0.25
    return pd.Series(raw, index=close.index).clip(-100.0, 100.0).fillna(0.0)


# -----------------------------------------------------------------------------
# Composite + classification
# -----------------------------------------------------------------------------
@dataclass
class ScoreResult:
    composite: pd.Series             # final composite score per bar
    per_market: Dict[str, pd.Series] # asset score per market
    contributions: pd.DataFrame      # per-market weighted contribution
    label: pd.Series                 # classification label
    confidence: pd.Series            # confidence %


def composite_score(
    data: Dict[str, pd.DataFrame],
    config: Config
) -> ScoreResult:
    """
    Compute the DE40 prediction composite for every bar in `data['de40']`.

    All inputs are aligned to a common index (the data layer is responsible
    for that).  Returns a `ScoreResult` carrying the composite, per-market
    scores, contributions and the 5-bucket label/confidence.
    """
    weights = config.normalized_weights()
    per_market: Dict[str, pd.Series] = {}
    contributions: Dict[str, pd.Series] = {}

    for mkt, df in data.items():
        if mkt not in weights:
            continue
        close = df["Close"]
        score = asset_score(close, config.periods)
        if mkt in NEGATIVE_CORRELATIONS:
            score = -score
        per_market[mkt] = score
        contributions[mkt] = score * weights[mkt]

    contrib_df = pd.DataFrame(contributions)
    composite = contrib_df.sum(axis=1).clip(-100.0, 100.0).rename("composite")

    label, confidence = _classify(composite)
    return ScoreResult(
        composite=composite,
        per_market=per_market,
        contributions=contrib_df,
        label=label,
        confidence=confidence,
    )


def _classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    5-bucket classification with confidence per the spec.
    """
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(62.5, index=score.index, dtype=float)

    label = label.mask(score >  70, "Strong Bullish")
    label = label.mask((score >  40) & (score <=  70), "Bullish")
    label = label.mask((score >= -40) & (score <=  40), "Neutral")
    label = label.mask((score >= -70) & (score <  -40), "Bearish")
    label = label.mask(score <  -70, "Strong Bearish")

    conf = conf.mask(score >  70, 95.0)
    conf = conf.mask((score >  40) & (score <=  70), 82.5)
    conf = conf.mask((score >= -40) & (score <=  40), 62.5)
    conf = conf.mask((score >= -70) & (score <  -40), 82.5)
    conf = conf.mask(score <  -70, 95.0)
    return label, conf


# -----------------------------------------------------------------------------
# Forecast horizons
# -----------------------------------------------------------------------------
def forecast_score(close: pd.Series, ema_fast: int, ema_slow: int, rsi_len: int) -> pd.Series:
    """
    Single horizon score: 70% trend (EMA fast/slow) + 30% momentum (RSI),
    clamped to [-100, +100].
    """
    e1 = ema(close, ema_fast)
    e2 = ema(close, ema_slow)
    t = np.where(e1 > e2, 100.0, np.where(e1 < e2, -100.0, 0.0))
    t_series = pd.Series(t, index=close.index).fillna(0.0)

    r = rsi(close, rsi_len)
    pos = ((r - 60.0) * (100.0 / 40.0)).clip(0.0, 100.0)
    neg = -((40.0 - r) * (100.0 / 40.0)).clip(0.0, 100.0)
    soft = (r - 50.0) * 5.0 * 0.10
    m_series = pd.Series(np.where(r > 60, pos, np.where(r < 40, neg, soft)),
                         index=close.index).fillna(0.0)

    return (t_series * 0.70 + m_series * 0.30).clip(-100.0, 100.0)


def forecasts(data: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.Series]:
    """Compute the three forecast horizon scores for DE40."""
    close = data["de40"]["Close"]
    out: Dict[str, pd.Series] = {}
    for name, cfg in config.forecasts.items():
        out[name] = forecast_score(
            close,
            ema_fast=cfg["ema_fast"],
            ema_slow=cfg["ema_slow"],
            rsi_len=cfg["rsi"],
        )
    return out


def forecast_classify(score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Per-horizon label + confidence in [50, 95]%."""
    label = pd.Series("Neutral", index=score.index, dtype=object)
    conf  = pd.Series(60.0, index=score.index, dtype=float)

    bull = score >  40
    bear = score < -40

    label = label.mask(bull, "Bullish")
    label = label.mask(bear, "Bearish")

    conf = conf.mask(
        bull,
        75.0 + ((score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0,
    )
    conf = conf.mask(
        bear,
        75.0 + ((-score - 40.0) / 60.0).clip(0.0, 1.0) * 20.0,
    )
    neutral = ~bull & ~bear
    conf = conf.mask(
        neutral,
        50.0 + (1.0 - score.abs() / 40.0).clip(0.0, 1.0) * 25.0,
    )
    return label, conf


# -----------------------------------------------------------------------------
# Market regime (Risk-On / Risk-Off / Transition)
# -----------------------------------------------------------------------------
def market_regime(per_market: Dict[str, pd.Series]) -> pd.Series:
    """
    Risk-On  : S&P  > 20  AND VIX < -10  AND Gold < -10
    Risk-Off : S&P  < -20 AND VIX >  10  AND Gold >  10
    Else     : Transition

    VIX and Gold in `per_market` are already sign-inverted (i.e. positive
    means risk-off), so we use them directly.
    """
    sp  = per_market["sp500"]
    vix = per_market["vix"]
    gld = per_market["gold"]

    on  = (sp >  20) & (vix < -10) & (gld < -10)
    off = (sp < -20) & (vix >  10) & (gld >  10)
    regime = pd.Series("Transition", index=sp.index, dtype=object)
    regime = regime.mask(on,  "Risk-On")
    regime = regime.mask(off, "Risk-Off")
    return regime


# -----------------------------------------------------------------------------
# Institutional flow meter  (0-100)
# -----------------------------------------------------------------------------
def flow_meter(
    composite: pd.Series,
    vix_close: pd.Series,
    dxy_close: pd.Series,
    nasdaq_score: pd.Series
) -> pd.Series:
    """
    0-100 gauge blending 50% composite, 20% inverted VIX, 15% inverted DXY,
    15% Nasdaq.  Higher = more institutional-friendly.
    """
    composite_n = (composite + 100.0) / 2.0

    vix_v = vix_close.clip(lower=0.0).fillna(15.0)
    vix_n  = ((1.0 - vix_v / 40.0).clip(0.0, 1.0)) * 100.0

    dxy_v = dxy_close.clip(lower=0.0).fillna(100.0)
    dxy_n = ((1.0 - (dxy_v - 90.0) / 20.0).clip(0.0, 1.0)) * 100.0

    nasdaq_n = (nasdaq_score + 100.0) / 2.0

    raw = (composite_n * 0.50 + vix_n * 0.20 + dxy_n * 0.15 + nasdaq_n * 0.15)
    return raw.clip(0.0, 100.0).fillna(50.0)
