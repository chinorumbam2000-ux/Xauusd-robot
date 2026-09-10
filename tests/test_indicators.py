import numpy as np
import pandas as pd
import pytest

from xauusd_robot.indicators import atr, body_range_ratio, ema, williams_percent_r


def test_ema_matches_manual_recursion():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(values, 3)
    alpha = 2 / (3 + 1)
    manual = values.iloc[0]
    for v in values.iloc[1:]:
        manual = alpha * v + (1 - alpha) * manual
    assert result.iloc[-1] == pytest.approx(manual)
    # warm-up bars must be NaN so no decision is taken on an unseeded EMA
    assert result.iloc[:2].isna().all()


def test_atr_uses_wilder_smoothing_and_true_range():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5])
    result = atr(high, low, close, period=2)
    # TR = [1.0, max(1.0, 1.5, 0.5) = 1.5, max(1.0, 1.5, 0.5) = 1.5]
    assert np.isnan(result.iloc[0])  # no ATR before the period is filled
    assert result.iloc[-1] == pytest.approx(1.5, abs=0.25)


def test_williams_percent_r_bounds_and_formula():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0])
    close = pd.Series([9.0, 10.0, 11.0, 13.0])
    result = williams_percent_r(high, low, close, period=3)
    hh, ll, c = 13.0, 10.0, 13.0
    expected = (hh - c) / (hh - ll) * -100.0
    assert result.iloc[-1] == pytest.approx(expected)
    assert result.dropna().between(-100, 0).all()


def test_williams_percent_r_extremes():
    """Close at the top of the range is 0 (overbought), at the bottom -100."""
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([0.0, 0.0, 0.0])
    top = williams_percent_r(high, low, pd.Series([10.0, 10.0, 10.0]), period=3)
    bottom = williams_percent_r(high, low, pd.Series([0.0, 0.0, 0.0]), period=3)
    assert top.iloc[-1] == pytest.approx(0.0)
    assert bottom.iloc[-1] == pytest.approx(-100.0)


def test_body_range_ratio():
    ratio = body_range_ratio(
        pd.Series([100.0]), pd.Series([101.0]), pd.Series([99.0]), pd.Series([100.6])
    )
    assert ratio.iloc[0] == pytest.approx(0.6 / 2.0)
