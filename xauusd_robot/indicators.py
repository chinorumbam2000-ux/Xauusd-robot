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


def body_range_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """|Close - Open| / (High - Low), the push-candle strength metric (Section 4.6)."""
    rng = (high - low).replace(0.0, np.nan)
    return (close - open_).abs() / rng


def add_core_indicators(df: pd.DataFrame, ema_period: int, atr_period: int, wpr_period: int) -> pd.DataFrame:
    """Return a copy of ``df`` (columns open/high/low/close) with EMA/ATR/WPR/body-ratio added."""
    out = df.copy()
    out["ema"] = ema(out["close"], ema_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)
    out["wpr"] = williams_percent_r(out["high"], out["low"], out["close"], wpr_period)
    out["body_range_ratio"] = body_range_ratio(out["open"], out["high"], out["low"], out["close"])
    out["bullish"] = out["close"] > out["open"]
    out["bearish"] = out["close"] < out["open"]
    return out
