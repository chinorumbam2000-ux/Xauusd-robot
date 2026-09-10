"""XAUUSD deterministic rule-based trading robot (Strategy Specification v1.1).

This package implements the research/backtesting layer described in the
Forex Robot XAUUSD Blueprint v1.1: a six-timeframe EMA200 regime filter,
an M5 zone engine (Fair Value Gap, Order Block, Support/Resistance),
Williams %R(49) momentum confirmation, a two push-candle entry trigger,
dynamic stop-distance based position sizing, and a full safety /
circuit-breaker layer.

The package is deliberately deterministic and free of machine learning,
matching the blueprint's design principle of an explainable, rule-based
baseline strategy (v1.1).
"""

__version__ = "1.1.0"
