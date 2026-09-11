"""What happens if entries move from M5 to M1?

Three things change at once, and they do not all point the same way.

Granularity. M1 gives five times as many bars, so more zones form and more
setups arm. That is the appeal.

Spread relative to volatility. ATR falls roughly with the square root of the
timeframe, so M1 ATR is about 45% of M5 ATR while the spread is unchanged. The
Section 5.5 filter caps spread at 0.15 x ATR, and structural stops shrink too,
so the same spread eats a larger share of every trade's risk.

Rule windows. Every "bars" parameter silently changes meaning: a 96-bar zone
expiry is 8 hours on M5 and 96 minutes on M1; an 8-bar setup expiry is 40
minutes against 8. Both readings are tested -- the parameters left alone, and
the parameters scaled by 5 so each rule covers the same wall-clock span.

History is the binding constraint: 100,000 M1 bars is only ~101 days, and the
H4 EMA200 consumes ~48,000 of them warming up.

    python scripts/m1_entries.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import COMBINATION_F, SYMBOL_SPECS, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402
from xauusd_robot.indicators import atr as atr_ind  # noqa: E402
from xauusd_robot.metrics import compute_metrics  # noqa: E402

from experiments import run  # noqa: E402

#: Parameters whose unit is BARS, and therefore whose meaning changes with the
#: execution timeframe.
BAR_PARAMS = ("setup_expiry_bars", "zone_max_age_bars", "cooldown_bars",
              "sr_lookback", "wpr_max_lead_bars", "reaction_max_bars")


def spread_profile(raw: pd.DataFrame, spec, label: str) -> dict:
    a = atr_ind(raw["high"], raw["low"], raw["close"], 14)
    spread = raw["spread"].astype(float) * spec.tick_size if "spread" in raw else None
    ratio = (spread / a).dropna()
    return {
        "data": label,
        "bars": len(raw),
        "days": (raw.index[-1] - raw.index[0]).days,
        "atr_median": round(float(a.median()), 3),
        "spread_median": round(float(spread.median()), 3),
        "spread/atr": round(float(ratio.median()), 4),
        "bars_over_0.15": f"{(ratio > 0.15).mean() * 100:.1f}%",
    }


def evaluate(raw, cfg, balance, label, start=None):
    res = run(raw, cfg, balance)
    trades = res.trades
    equity = res.equity
    if start is not None:
        # The equity frame carries a RangeIndex with a separate time column, so
        # slicing on the index silently does nothing and leaves whole-run
        # drawdown and return figures attached to a windowed row.
        if not trades.empty:
            trades = trades[trades["entry_time"] >= start]
        if "time" in equity.columns:
            equity = equity[pd.to_datetime(equity["time"]) >= start].reset_index(drop=True)
    m = compute_metrics(trades, equity, balance, cfg)
    days = (raw.index[-1] - raw.index[0]).days or 1
    f = res.metrics.get("funnel", {})
    return {
        "variant": label,
        "trades": m["trades"],
        "per_week": round(m["trades"] / (days / 7.0), 2),
        "win_%": round(m["win_rate"], 1),
        "exp_R": round(m["expectancy_r"], 3),
        "PF": round(m["profit_factor"], 2) if m["profit_factor"] == m["profit_factor"] else None,
        "net_%": round(m["net_return_percent"], 1),
        "maxDD_%": round(m["max_drawdown_percent"], 2),
        "armed": f.get("setups_armed", 0),
        "spread_rej": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m1", default="data/XAUUSD_M1_live.csv")
    parser.add_argument("--m5", default="data/XAUUSD_M5_live.csv")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=2.0)
    args = parser.parse_args()

    spec = SYMBOL_SPECS["XAUUSD"]
    m1 = load_m5_csv(args.m1)
    m5 = load_m5_csv(args.m5)

    print("=== Spread versus volatility ===")
    print(pd.DataFrame([
        spread_profile(m5, spec, "M5"),
        spread_profile(m1, spec, "M1"),
    ]).to_string(index=False))
    print("\n  The 0.15 cap is the Section 5.5 filter. A higher ratio means the")
    print("  spread is a larger share of every stop, and of every R earned.\n")

    base = dict(broker=spec, risk_percent_initial_balance=args.risk, **COMBINATION_F)

    # M5 restricted to the same calendar window the M1 file covers, so the
    # comparison is not simply two different market periods.
    overlap_start = m1.index[0]
    cfg_m5 = replace(StrategyConfig(**base), execution_tf="M5",
                     regime_timeframes=("H4", "H1", "M30", "M5"))
    rows = [evaluate(m5, cfg_m5, args.balance, "M5 entries (current), full 517d")]
    rows.append(evaluate(m5, cfg_m5, args.balance,
                         f"M5 entries, same {(m1.index[-1] - overlap_start).days}d window",
                         start=overlap_start))

    cfg_m1 = replace(StrategyConfig(**base), execution_tf="M1",
                     regime_timeframes=("H4", "H1", "M30", "M1"))
    rows.append(evaluate(m1, cfg_m1, args.balance, "M1 entries, parameters unchanged"))

    # Scale every bar-denominated window by 5 so each rule spans the same
    # wall-clock time it did on M5. Values already set by combination F are
    # scaled from those, not from the blueprint defaults.
    scaled = {k: int(getattr(cfg_m1, k) * 5) for k in BAR_PARAMS}
    cfg_m1_scaled = replace(cfg_m1, **scaled)
    rows.append(evaluate(m1, cfg_m1_scaled, args.balance, "M1 entries, bar windows x5"))

    print("=== Entries on M1 versus M5 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  per_week normalises for the different history lengths.")
    print(f"  M1 file covers {(m1.index[-1] - m1.index[0]).days} days; the H4 EMA200 consumes")
    print("  roughly 48,000 M1 bars warming up, so the tradeable part is far shorter.")


if __name__ == "__main__":
    main()
