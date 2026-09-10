"""Section 6.2 / 12: diagnostic charts for the Python research layer.

Equity curve, drawdown curve, R distribution and per-zone-type performance.
Matplotlib is imported lazily so the core engine has no plotting dependency.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_run(result, out_dir: str, prefix: str = "") -> Optional[str]:
    """Write a four-panel diagnostic figure. Returns the file path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    equity, trades = result.equity, result.trades
    if equity.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"XAUUSD v1.1 baseline -- risk {result.config.risk_percent_initial_balance}% of initial balance",
        fontsize=12,
    )

    ax = axes[0][0]
    ax.plot(pd.to_datetime(equity["time"]), equity["equity"], linewidth=1.0, color="#1f77b4")
    ax.axhline(result.initial_balance, color="#999", linewidth=0.8, linestyle="--")
    _style(ax, "Equity curve", ylabel="account currency")

    ax = axes[0][1]
    ax.fill_between(
        pd.to_datetime(equity["time"]), equity["drawdown"] * 100.0, 0,
        color="#d62728", alpha=0.35, linewidth=0,
    )
    ax.axhline(
        result.config.max_peak_equity_drawdown * 100.0,
        color="#d62728", linewidth=0.9, linestyle="--",
        label=f"{result.config.max_peak_equity_drawdown:.0%} lockout",
    )
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Drawdown from running equity peak", ylabel="%")

    ax = axes[1][0]
    if not trades.empty:
        ax.hist(trades["r_multiple"], bins=20, color="#2ca02c", alpha=0.75)
        ax.axvline(0, color="#333", linewidth=0.8)
    _style(ax, "Trade outcome distribution", xlabel="R multiple", ylabel="trades")

    ax = axes[1][1]
    if not trades.empty and "zone_type" in trades.columns:
        totals = trades.groupby("zone_type")["r_multiple"].sum()
        ax.bar(totals.index, totals.values, color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="#333", linewidth=0.8)
    _style(ax, "Total R by zone type", ylabel="R")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"{prefix}report.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
