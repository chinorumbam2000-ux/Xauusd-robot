"""Section 3.1: Multi-Timeframe EMA200 Trend Filter.

BUY regime  -> closed price > EMA200 on D1, H4, H1, M30, M15, M5 (all six)
SELL regime -> closed price < EMA200 on all six
Anything else -> MIXED (no new trade)
"""
from __future__ import annotations

import enum

import numpy as np
import pandas as pd


class Regime(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    MIXED = "MIXED"


def compute_regime(regime_frame: pd.DataFrame, timeframes) -> pd.Series:
    """Vectorized regime computation over an M5-aligned frame.

    ``regime_frame`` must contain ``f"{tf}_close"`` and ``f"{tf}_ema"``
    columns for every timeframe in ``timeframes`` (see
    :func:`xauusd_robot.data.build_regime_frame`).
    """
    above = pd.DataFrame(index=regime_frame.index)
    below = pd.DataFrame(index=regime_frame.index)
    valid = pd.Series(True, index=regime_frame.index)
    for tf in timeframes:
        close_col, ema_col = f"{tf}_close", f"{tf}_ema"
        valid &= regime_frame[close_col].notna() & regime_frame[ema_col].notna()
        above[tf] = regime_frame[close_col] > regime_frame[ema_col]
        below[tf] = regime_frame[close_col] < regime_frame[ema_col]

    all_above = above.all(axis=1)
    all_below = below.all(axis=1)

    out = pd.Series(Regime.MIXED, index=regime_frame.index, dtype=object)
    out[valid & all_above] = Regime.BUY
    out[valid & all_below] = Regime.SELL
    out[~valid] = Regime.MIXED
    return out
