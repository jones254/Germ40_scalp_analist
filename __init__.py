"""
Germany 40 Institutional Prediction Engine
============================================

A self-contained, multi-market macro prediction engine for DE40 (DAX).

Architecture:
    data.py       - yfinance / Twelve Data abstraction
    indicators.py - EMA, RSI, ROC, ATR (Pine-equivalent implementations)
    scoring.py    - 7-market composite + forecasts (port of the Pine Script)
    backtest.py   - historical backtest engine
    config.py     - weights, periods, data-source selection
"""

__version__ = "1.0.0"
