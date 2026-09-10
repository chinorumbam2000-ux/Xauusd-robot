"""Shared helpers for building deterministic hand-made bar series."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import pytest


def make_bars(rows: Sequence[Sequence[float]], start: str = "2024-01-01 00:00", atr: float | Iterable[float] | None = None, wpr: Iterable[float] | None = None) -> pd.DataFrame:
    """Build an M5 OHLC frame from ``(open, high, low, close)`` tuples.

    ``atr`` and ``wpr`` may be supplied directly so zone / momentum rules can
    be tested against exact thresholds instead of warm-up dependent values.
    """
    index = pd.date_range(start=start, periods=len(rows), freq="5min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    df.index.name = "time"
    if atr is not None:
        df["atr"] = atr if not isinstance(atr, (int, float)) else float(atr)
    if wpr is not None:
        df["wpr"] = list(wpr)
    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]
    return df


@pytest.fixture
def config():
    from xauusd_robot.config import StrategyConfig

    return StrategyConfig()
