"""Does the strategy work on anything other than gold?

The rules are all ATR-relative, so they carry across instruments without
rescaling. What does not carry is the relationship between spread and
volatility: gold's M5 range dwarfs its spread, while a forex major's does not.
The 1:3 payoff is what makes that decisive -- a stop only a few pips wide gives
back a large fraction of its risk to the spread on entry.

Each symbol is backtested with its own broker specification and its own
recorded per-bar spread, then walk-forward tested. Nothing is assumed to
transfer from the XAUUSD result.

    python scripts/symbol_comparison.py
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import COMBINATION_F, SYMBOL_SPECS, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402
from xauusd_robot.indicators import atr as atr_indicator  # noqa: E402

from experiments import run, walk_forward_score  # noqa: E402


def spread_profile(raw: pd.DataFrame, spec) -> dict:
    """Spread relative to volatility -- the number that decides viability."""
    atr = atr_indicator(raw["high"], raw["low"], raw["close"], 14)
    if "spread" in raw.columns:
        spread_price = raw["spread"].astype(float) * spec.tick_size
    else:
        spread_price = pd.Series(spec.default_spread_points * spec.tick_size, index=raw.index)
    ratio = (spread_price / atr).dropna()
    return {
        "atr_median": float(atr.median()),
        "spread_median": float(spread_price.median()),
        "spread_atr_median": round(float(ratio.median()), 4),
        "spread_atr_p95": round(float(ratio.quantile(0.95)), 4),
        "bars_over_limit_%": round(float((ratio > 0.15).mean() * 100), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbols", nargs="*", default=["XAUUSD", "GBPUSD", "EURUSD", "USDJPY", "EURGBP"])
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=2.0)
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--test-bars", type=int, default=14000)
    args = parser.parse_args()

    rows, profiles = [], []
    for symbol in args.symbols:
        path = os.path.join(args.data_dir, f"{symbol}_M5_live.csv")
        if not os.path.exists(path):
            print(f"  {symbol}: no data at {path}, skipping")
            continue
        raw = load_m5_csv(path)
        spec = SYMBOL_SPECS[symbol]

        profile = spread_profile(raw, spec)
        profiles.append({"symbol": symbol, **profile})

        config = replace(
            StrategyConfig(broker=spec), regime_timeframes=("H4", "H1", "M30", "M15", "M5"),
            risk_percent_initial_balance=args.risk, **COMBINATION_F,
        )
        result = run(raw, config, args.balance)
        m = result.metrics
        funnel = m.get("funnel", {})
        wf = walk_forward_score(raw, config, args.balance, args.train_bars, args.test_bars)

        rejects = result.events
        spread_rejects = 0
        if not rejects.empty and "reason" in rejects.columns:
            spread_rejects = int(rejects["reason"].astype(str).str.contains("spread").sum())

        rows.append({
            "symbol": symbol,
            "bars": len(raw),
            "trades": m["trades"],
            "win_%": round(m["win_rate"], 1),
            "exp_R": round(m["expectancy_r"], 3),
            "PF": round(m["profit_factor"], 2) if m["profit_factor"] == m["profit_factor"] else None,
            "net_%": round(m["net_return_percent"], 1),
            "maxDD_%": round(m["max_drawdown_percent"], 2),
            "armed": funnel.get("setups_armed", 0),
            "spread_rej": spread_rejects,
            "wf_win": f"{wf['profitable']}/{wf['windows']}",
            "wf_R": wf["wf_total_r"],
            "wf_worst": wf["worst_r"],
        })
        print(f"  ran {symbol}")

    print("\n=== Spread versus volatility (the viability test) ===")
    print(pd.DataFrame(profiles).to_string(index=False))
    print("\n  spread_atr_median is the Section 5.5 filter input; the cap is 0.15.")
    print("  bars_over_limit_% is how often the market is untradeable by that rule.")

    print(f"\n=== Backtest per symbol (combination F, {args.risk}% risk) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  spread_rej counts setups that reached entry and were refused on spread.")


if __name__ == "__main__":
    main()
