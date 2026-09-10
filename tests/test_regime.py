"""Section 3.1 + the "EMA alignment" / "No look-ahead" acceptance tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xauusd_robot.config import StrategyConfig
from xauusd_robot.data import align_to_m5_close, build_regime_frame, build_timeframe_indicators, resample_ohlc
from xauusd_robot.regime import Regime, compute_regime

TFS = ("D1", "H4", "H1", "M30", "M15", "M5")


def _frame(rows):
    index = pd.date_range("2024-01-01", periods=len(rows), freq="5min")
    data = {}
    for tf, values in rows.items():
        data[f"{tf}_close"] = [v[0] for v in values]
        data[f"{tf}_ema"] = [v[1] for v in values]
    return pd.DataFrame(data, index=index[: len(next(iter(rows.values())))])


def test_buy_regime_requires_all_six_timeframes_above_ema():
    frame = _frame({tf: [(101.0, 100.0)] for tf in TFS})
    assert compute_regime(frame, TFS).iloc[0] is Regime.BUY


def test_single_violating_timeframe_forces_mixed():
    rows = {tf: [(101.0, 100.0)] for tf in TFS}
    rows["H4"] = [(99.0, 100.0)]
    assert compute_regime(_frame(rows), TFS).iloc[0] is Regime.MIXED


def test_sell_regime_requires_all_six_below():
    frame = _frame({tf: [(99.0, 100.0)] for tf in TFS})
    assert compute_regime(frame, TFS).iloc[0] is Regime.SELL


def test_price_exactly_on_ema_is_not_alignment():
    """"strictly above" / "strictly below" -- equality is MIXED."""
    rows = {tf: [(101.0, 100.0)] for tf in TFS}
    rows["M15"] = [(100.0, 100.0)]
    assert compute_regime(_frame(rows), TFS).iloc[0] is Regime.MIXED


def test_missing_higher_timeframe_ema_is_mixed_not_a_guess():
    rows = {tf: [(101.0, 100.0)] for tf in TFS}
    rows["D1"] = [(101.0, np.nan)]
    assert compute_regime(_frame(rows), TFS).iloc[0] is Regime.MIXED


# ------------------------------------------------------- no look-ahead ----
def _synthetic_m5(bars=1200):
    index = pd.date_range("2024-01-01", periods=bars, freq="5min")
    close = pd.Series(np.linspace(2000, 2100, bars), index=index)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(2000.0), "high": close + 1.0, "low": close - 1.0, "close": close}
    )


def test_resampled_bar_close_time_is_open_time_plus_timeframe():
    m5 = _synthetic_m5(600)
    h1 = resample_ohlc(m5, "H1")
    assert (h1["close_time"] - h1.index == pd.Timedelta(hours=1)).all()


def test_higher_timeframe_value_is_never_from_an_unfinished_candle():
    """An M5 bar may only see H1 data whose bar closed at or before it."""
    m5 = _synthetic_m5(600)
    h1 = resample_ohlc(m5, "H1")
    m5_close = m5.index.to_series() + pd.Timedelta(minutes=5)
    aligned = align_to_m5_close(m5_close, h1, prefix="H1", columns=("close",))

    for ts, value in aligned["H1_close"].dropna().items():
        m5_bar_close = ts + pd.Timedelta(minutes=5)
        source = h1[h1["close"] == value]
        assert (source["close_time"] <= m5_bar_close).any()

    # the very first M5 bars have no completed H1 bar behind them
    assert aligned["H1_close"].iloc[0:11].isna().all()


def test_regime_frame_uses_previous_closed_higher_timeframe_bar():
    m5 = _synthetic_m5(900)
    config = StrategyConfig(ema_period=5, atr_period=3, wpr_period=5)
    tf_frames = build_timeframe_indicators(m5, config)
    frame = build_regime_frame(m5, tf_frames, config)

    h1 = tf_frames["H1"]
    probe = m5.index[400]
    expected = h1[h1["close_time"] <= probe + pd.Timedelta(minutes=5)]["close"].iloc[-1]
    assert frame.loc[probe, "H1_close"] == pytest.approx(expected)
