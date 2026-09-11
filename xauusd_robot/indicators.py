"""Deterministic indicator calculations shared across timeframes.

All functions operate on closed-bar OHLC data only and are careful not to
introduce look-ahead: every value at index ``i`` depends only on bars
``0..i``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average, alpha = 2 / (period + 1)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (the conventional ATR)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing == EWM with alpha = 1/period
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def williams_percent_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 49) -> pd.Series:
    """Williams %R, conventional -20 (overbought) to -80 (oversold) bounds.

    %R = (HighestHigh(period) - Close) / (HighestHigh(period) - LowestLow(period)) * -100
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    rng = (highest_high - lowest_low).replace(0.0, np.nan)
    wpr = (highest_high - close) / rng * -100.0
    return wpr


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Standard MACD. Returns (line, signal, histogram).

    The histogram's sign change is the signal-line cross, which is the closest
    structural analogue to Williams %R leaving an extreme: momentum was against
    the trade, and has now turned in its favour.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def body_range_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """|Close - Open| / (High - Low), the push-candle strength metric (Section 4.6)."""
    rng = (high - low).replace(0.0, np.nan)
    return (close - open_).abs() / rng


def add_core_indicators(df: pd.DataFrame, ema_period: int, atr_period: int, wpr_period: int,
                        macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """Return a copy of ``df`` (columns open/high/low/close) with EMA/ATR/WPR/body-ratio added."""
    out = df.copy()
    out["ema"] = ema(out["close"], ema_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)
    out["wpr"] = williams_percent_r(out["high"], out["low"], out["close"], wpr_period)
    _, _, out["macd_hist"] = macd(out["close"], macd_fast, macd_slow, macd_signal)
    out["body_range_ratio"] = body_range_ratio(out["open"], out["high"], out["low"], out["close"])
    out["bullish"] = out["close"] > out["open"]
    out["bearish"] = out["close"] < out["open"]
    return out
