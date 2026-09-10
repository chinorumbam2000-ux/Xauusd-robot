"""Section 9.3: mandatory performance metrics, plus Monte Carlo sequence testing.

A 1:3 payoff has a theoretical break-even win rate of 25% before costs
(Section 9.4). Real break-even is higher once spread, slippage and gaps
are included, so ``break_even_win_rate`` is reported alongside the
achieved win rate to keep that margin visible.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def max_drawdown(equity: pd.Series) -> Dict[str, float]:
    if equity.empty:
        return {"max_drawdown_money": 0.0, "max_drawdown_percent": 0.0}
    running_peak = equity.cummax()
    dd_money = (running_peak - equity)
    dd_pct = dd_money / running_peak.replace(0, np.nan)
    return {
        "max_drawdown_money": float(dd_money.max()),
        "max_drawdown_percent": float(dd_pct.max() * 100.0) if not dd_pct.isna().all() else 0.0,
    }


def losing_streaks(r_values: pd.Series) -> Dict[str, object]:
    streaks: List[int] = []
    current = 0
    for r in r_values:
        if r < 0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    distribution = pd.Series(streaks).value_counts().sort_index().to_dict() if streaks else {}
    return {
        "max_consecutive_losses": max(streaks) if streaks else 0,
        "losing_streak_distribution": {int(k): int(v) for k, v in distribution.items()},
    }


def _group_stats(trades: pd.DataFrame, column: str) -> Dict[str, Dict[str, float]]:
    if trades.empty or column not in trades.columns:
        return {}
    out = {}
    for key, group in trades.groupby(column):
        wins = group[group["r_multiple"] > 0]
        out[str(key)] = {
            "trades": int(len(group)),
            "win_rate": _safe_div(len(wins), len(group)) * 100.0,
            "expectancy_r": float(group["r_multiple"].mean()),
            "total_r": float(group["r_multiple"].sum()),
            "pnl_money": float(group["pnl_money"].sum()),
        }
    return out


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, initial_balance: float, config) -> Dict:
    metrics: Dict = {
        "initial_balance": initial_balance,
        "risk_percent": config.risk_percent_initial_balance,
        "reward_risk": config.reward_risk,
        "break_even_win_rate": 100.0 / (1.0 + config.reward_risk),
        "trades": int(len(trades)),
    }

    if trades.empty:
        metrics.update({
            "final_balance": initial_balance, "net_return_money": 0.0, "net_return_percent": 0.0,
            "win_rate": 0.0, "expectancy_r": 0.0, "total_r": 0.0, "profit_factor": float("nan"),
            "max_drawdown_money": 0.0, "max_drawdown_percent": 0.0,
            "max_consecutive_losses": 0, "trades_per_month": 0.0, "target_reached": False,
        })
        return metrics

    wins = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    gross_profit = float(wins["pnl_money"].sum())
    gross_loss = float(-losses["pnl_money"].sum())
    final_balance = float(trades["balance_after"].iloc[-1])

    span_days = (trades["exit_time"].max() - trades["entry_time"].min()).total_seconds() / 86400.0
    months = span_days / 30.4375 if span_days > 0 else 0.0

    metrics.update({
        "final_balance": final_balance,
        "net_return_money": final_balance - initial_balance,
        "net_return_percent": (final_balance / initial_balance - 1.0) * 100.0,
        "win_rate": _safe_div(len(wins), len(trades)) * 100.0,
        "expectancy_r": float(trades["r_multiple"].mean()),
        "total_r": float(trades["r_multiple"].sum()),
        "profit_factor": _safe_div(gross_profit, gross_loss),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_win_r": float(wins["r_multiple"].mean()) if len(wins) else 0.0,
        "average_loss_r": float(losses["r_multiple"].mean()) if len(losses) else 0.0,
        "trades_per_month": _safe_div(len(trades), months) if months else float(len(trades)),
        "target_reached": bool(final_balance >= config.target_multiplier * initial_balance),
        "by_direction": _group_stats(trades, "direction"),
        "by_zone_type": _group_stats(trades, "zone_type"),
        "by_confluence": _group_stats(trades, "confluence_score"),
        "by_session": _group_stats(trades, "session"),
        "by_exit_reason": _group_stats(trades, "exit_reason"),
    })
    metrics.update(losing_streaks(trades["r_multiple"]))
    if not equity.empty:
        metrics.update(max_drawdown(equity["equity"]))
    else:
        metrics.update(max_drawdown(trades["balance_after"]))
    return metrics


def monte_carlo_target_probability(
    r_values: np.ndarray,
    initial_balance: float,
    risk_money: float,
    target_multiplier: float = 10.0,
    max_drawdown_lock: float = 0.15,
    simulations: int = 2000,
    trades_per_path: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Section 9.3 "Target probability": how often a resampled trade sequence
    reaches the 10x target before hitting the peak-equity drawdown lock.

    Trades are resampled with replacement -- this destroys serial ordering
    on purpose and is used only as a sequence-risk stress test, never as a
    substitute for chronological validation (Section 9.2).
    """
    if len(r_values) == 0:
        return {"target_probability": 0.0, "lock_probability": 0.0, "neither": 1.0}

    rng = np.random.default_rng(seed)
    target_balance = initial_balance * target_multiplier
    hits, locks = 0, 0
    for _ in range(simulations):
        draws = rng.choice(r_values, size=trades_per_path, replace=True)
        balance = initial_balance
        peak = initial_balance
        for r in draws:
            balance += r * risk_money
            peak = max(peak, balance)
            if balance >= target_balance:
                hits += 1
                break
            if peak > 0 and (1.0 - balance / peak) >= max_drawdown_lock:
                locks += 1
                break
    return {
        "target_probability": hits / simulations,
        "lock_probability": locks / simulations,
        "neither": 1.0 - (hits + locks) / simulations,
    }
