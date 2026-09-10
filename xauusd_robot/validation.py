"""Sections 9.2 / 10: chronological test segmentation and robustness sweeps.

"Never randomly shuffle time-series data for the core strategy
validation." Every split here is strictly chronological. Parameter sweeps
report the whole surface so a *plateau* can be judged, rather than
returning a single best-performing value.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .backtester import Backtester
from .config import RISK_TEST_MATRIX, StrategyConfig


def chronological_split(m5: pd.DataFrame, fractions: Tuple[float, float, float] = (0.5, 0.25, 0.25)) -> Dict[str, pd.DataFrame]:
    """Split into in-sample / validation / out-of-sample, in time order."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1.0")
    n = len(m5)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return {
        "in_sample": m5.iloc[:a],
        "validation": m5.iloc[a:b],
        "out_of_sample": m5.iloc[b:],
    }


def walk_forward_windows(m5: pd.DataFrame, train_bars: int, test_bars: int) -> List[Dict[str, pd.DataFrame]]:
    """Rolling chronological train/test windows (Section 9.2 walk-forward)."""
    windows = []
    start = 0
    while start + train_bars + test_bars <= len(m5):
        windows.append({
            "train": m5.iloc[start : start + train_bars],
            "test": m5.iloc[start + train_bars : start + train_bars + test_bars],
        })
        start += test_bars
    return windows


def run_backtest(m5: pd.DataFrame, config: StrategyConfig, initial_balance: float = 1000.0):
    return Backtester(m5, config, initial_balance).run()


def risk_matrix_comparison(
    m5: pd.DataFrame,
    base_config: StrategyConfig = StrategyConfig(),
    initial_balance: float = 1000.0,
    risks: Iterable[float] = RISK_TEST_MATRIX,
) -> pd.DataFrame:
    """Section 5.1 / 17.2: compare 0.5/1/2/3/5% risk as separate experiments."""
    rows = []
    for risk in risks:
        cfg = replace(base_config, risk_percent_initial_balance=risk)
        result = run_backtest(m5, cfg, initial_balance)
        m = result.metrics
        rows.append({
            "risk_percent": risk,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "expectancy_r": m["expectancy_r"],
            "profit_factor": m["profit_factor"],
            "net_return_percent": m["net_return_percent"],
            "max_drawdown_percent": m["max_drawdown_percent"],
            "max_consecutive_losses": m["max_consecutive_losses"],
            "final_balance": m["final_balance"],
            "target_reached": m["target_reached"],
        })
    return pd.DataFrame(rows)


def parameter_sweep(
    m5: pd.DataFrame,
    parameter: str,
    values: Iterable,
    base_config: StrategyConfig = StrategyConfig(),
    initial_balance: float = 1000.0,
) -> pd.DataFrame:
    """Vary one rule family at a time and report the full surface (Section 10).

    Selection rule: prefer a broad plateau of good performance across
    neighbouring values; a narrow spike is a common sign of overfitting.
    """
    rows = []
    for value in values:
        cfg = replace(base_config, **{parameter: value})
        result = run_backtest(m5, cfg, initial_balance)
        m = result.metrics
        rows.append({
            parameter: value,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "expectancy_r": m["expectancy_r"],
            "profit_factor": m["profit_factor"],
            "net_return_percent": m["net_return_percent"],
            "max_drawdown_percent": m["max_drawdown_percent"],
        })
    return pd.DataFrame(rows)


def spread_stress_test(
    m5: pd.DataFrame,
    multipliers: Iterable[float] = (1.0, 1.5, 2.0, 3.0),
    base_config: StrategyConfig = StrategyConfig(),
    initial_balance: float = 1000.0,
) -> pd.DataFrame:
    """Section 9.3 / 16: re-run under progressively worse execution assumptions."""
    rows = []
    for mult in multipliers:
        broker = replace(base_config.broker, default_spread_points=base_config.broker.default_spread_points * mult)
        cfg = replace(base_config, broker=broker)
        data = m5.copy()
        if "spread" in data.columns:
            data["spread"] = data["spread"] * mult
        result = run_backtest(data, cfg, initial_balance)
        m = result.metrics
        rows.append({
            "spread_multiplier": mult,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "expectancy_r": m["expectancy_r"],
            "net_return_percent": m["net_return_percent"],
            "max_drawdown_percent": m["max_drawdown_percent"],
        })
    return pd.DataFrame(rows)
