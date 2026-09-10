"""Confirmed swing-high / swing-low pivot detection (Section 4.3).

"Confirmed swing high: the pivot high is greater than the highs of the 2
closed bars immediately before it and the 2 bars immediately after it."
Symmetric for swing lows. Used both for Support/Resistance zones and for
the Order Block "close beyond the most recent confirmed swing" rule.

A pivot at bar index ``p`` only becomes knowable once ``right`` further
bars have closed (bar ``p + right``); :func:`last_confirmed_value` encodes
that lag explicitly so downstream code never leaks the future.
"""
from __future__ import annotations

import pandas as pd


def confirmed_pivots(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    """Return (pivot_high, pivot_low) boolean Series, indexed like the input.

    ``pivot_high[p] is True`` means bar ``p`` is a confirmed swing high --
    but that fact is only observable starting at bar ``p + right``.
    """
    pivot_high = pd.Series(True, index=high.index)
    pivot_low = pd.Series(True, index=low.index)
    for k in range(1, left + 1):
        pivot_high &= high > high.shift(k)
        pivot_low &= low < low.shift(k)
    for k in range(1, right + 1):
        pivot_high &= high > high.shift(-k)
        pivot_low &= low < low.shift(-k)
    return pivot_high.fillna(False), pivot_low.fillna(False)


def last_confirmed_value(is_pivot: pd.Series, price: pd.Series, right: int) -> pd.Series:
    """As-of-now last confirmed pivot price, with no look-ahead.

    At bar ``j`` this reports the most recent pivot price that was already
    confirmed by bar ``j`` (i.e. whose pivot bar was ``<= j - right``).
    """
    confirmed_now = is_pivot.shift(right).fillna(False)
    value_at_confirmation = price.shift(right).where(confirmed_now)
    return value_at_confirmation.ffill()
