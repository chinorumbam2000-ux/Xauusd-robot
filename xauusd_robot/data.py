"""Data loading and no-look-ahead multi-timeframe alignment.

Design Constraint (Blueprint Section 2): "The EA must avoid look-ahead
bias in all timeframes." All trading decisions must use the most recently
CLOSED candle on each timeframe (Section 3.1).

We implement this with the standard, auditable technique: resample the M5
base data into each higher timeframe (bars are timestamped by their OPEN
time), attach an explicit ``close_time`` to every bar, then use
``pd.merge_asof(..., direction="backward")`` keyed on ``close_time`` to
attach, to every M5 bar, only the higher-timeframe bar whose close_time is
at or before that M5 bar's own close_time. A higher-timeframe bar that is
still forming when the M5 bar closes is therefore never visible.
"""
from __future__ import annotations

from typing import Dict, Mapping

import pandas as pd

from .indicators import add_core_indicators

TF_FREQ: Mapping[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

REQUIRED_COLUMNS = ("open", "high", "low", "close")


def load_m5_csv(path: str, tz: str | None = None) -> pd.DataFrame:
    """Load M5 OHLC(V) data from CSV.

    Expects a ``time`` (or ``datetime``/``date``) column plus
    open/high/low/close and optionally volume/spread. Column names are
    matched case-insensitively.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in ("time", "datetime", "date") if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"{path}: could not find a time/datetime/date column")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    if tz is not None:
        df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    keep = list(REQUIRED_COLUMNS) + [c for c in ("volume", "spread") if c in df.columns]
    df = df[keep]
    return df.astype({c: "float64" for c in REQUIRED_COLUMNS})


def resample_ohlc(m5: pd.DataFrame, timeframe: str, base_tf: str = "M5") -> pd.DataFrame:
    """Resample base-timeframe bars up to ``timeframe``. Bars are indexed by OPEN time.

    ``base_tf`` is the execution timeframe the input already is; asking for it
    back is a no-op rather than a resample.
    """
    if timeframe == base_tf:
        out = m5[["open", "high", "low", "close"]].copy()
    else:
        rule = TF_FREQ[timeframe]
        agg = {
            "open": m5["open"].resample(rule, label="left", closed="left").first(),
            "high": m5["high"].resample(rule, label="left", closed="left").max(),
            "low": m5["low"].resample(rule, label="left", closed="left").min(),
            "close": m5["close"].resample(rule, label="left", closed="left").last(),
        }
        out = pd.DataFrame(agg).dropna(subset=["open", "high", "low", "close"])
    out.index.name = "open_time"
    freq = pd.tseries.frequencies.to_offset(TF_FREQ[timeframe])
    out["close_time"] = out.index + freq
    return out


def build_timeframe_indicators(m5: pd.DataFrame, config) -> Dict[str, pd.DataFrame]:
    """Resample to every regime timeframe and attach EMA/ATR/WPR indicators."""
    result: Dict[str, pd.DataFrame] = {}
    for tf in config.regime_timeframes:
        bars = resample_ohlc(m5, tf, config.execution_tf)
        enriched = add_core_indicators(
            bars, config.ema_period, config.atr_period, config.wpr_period
        )
        enriched["close_time"] = bars["close_time"]
        result[tf] = enriched
    return result


def align_to_m5_close(m5_close_time: pd.Series, higher_tf: pd.DataFrame, prefix: str, columns=("close", "ema")) -> pd.DataFrame:
    """Attach the most recently CLOSED higher-timeframe bar to each M5 timestamp.

    ``m5_close_time`` must be sorted ascending. Returns a frame aligned to
    the same index as ``m5_close_time`` with columns named
    ``f"{prefix}_{col}"``.
    """
    right = higher_tf.reset_index()[["close_time", *columns]].sort_values("close_time")
    left = pd.DataFrame({"close_time": m5_close_time.values}, index=m5_close_time.index)
    merged = pd.merge_asof(
        left.reset_index(names="_idx"),
        right,
        on="close_time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_idx")
    merged.index.name = m5_close_time.index.name
    merged = merged.rename(columns={col: f"{prefix}_{col}" for col in columns})
    return merged[[f"{prefix}_{col}" for col in columns]]


def build_regime_frame(m5_with_indicators: pd.DataFrame, tf_frames: Dict[str, pd.DataFrame], config) -> pd.DataFrame:
    """Build a single M5-indexed frame with each regime timeframe's closed
    price and EMA200 aligned with no look-ahead, ready for regime evaluation.
    """
    base_freq = TF_FREQ[config.execution_tf]
    m5_close_time = m5_with_indicators.index.to_series() + pd.tseries.frequencies.to_offset(base_freq)
    pieces = []
    for tf in config.regime_timeframes:
        aligned = align_to_m5_close(m5_close_time, tf_frames[tf], prefix=tf, columns=("close", "ema"))
        pieces.append(aligned)
    out = pd.concat(pieces, axis=1)
    return out
