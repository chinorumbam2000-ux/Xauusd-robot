"""Section 3.4 / 4.6: Two consecutive directional push candles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float

    @property
    def body_range_ratio(self) -> float:
        rng = self.high - self.low
        if rng <= 0:
            return 0.0
        return abs(self.close - self.open) / rng

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


def is_qualifying_push(candle: Candle, direction: str, push_body_ratio: float) -> bool:
    """A push candle must close in the trade direction with body/range >= threshold."""
    if direction == "BUY":
        return candle.is_bullish and candle.body_range_ratio >= push_body_ratio
    if direction == "SELL":
        return candle.is_bearish and candle.body_range_ratio >= push_body_ratio
    raise ValueError(f"unknown direction {direction!r}")


def push2_breaks_push1(push1: Candle, push2: Candle, direction: str) -> bool:
    """BUY: Push2 close > Push1 high. SELL: Push2 close < Push1 low."""
    if direction == "BUY":
        return push2.close > push1.high
    if direction == "SELL":
        return push2.close < push1.low
    raise ValueError(f"unknown direction {direction!r}")
