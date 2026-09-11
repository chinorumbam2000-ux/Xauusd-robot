"""Run the strategy on a synthetic index, where there is provably no edge.

Deriv's Step indices are constructed random walks: fixed-size steps, roughly
even odds either way. Nothing defends a level there, so an "order block" is
just a price the walk happened to visit twice. A strategy built on price
reacting at levels should therefore find nothing.

That makes them a null hypothesis with a known answer, and a sharper overfitting
check than walk-forward. Walk-forward asks whether an edge persists across time;
this asks whether the edge exists at all, on data that cannot contain one. If a
rule change makes the strategy profitable HERE, the change is fitting noise.

It also rules out the alternative explanation for a good backtest: that the 1:3
payoff structure or a bug in the harness is manufacturing apparent edge. If that
were happening, the synthetics would show fake profit too.

The expected result is a win rate at the payoff's geometric break-even --
25% for 1:3 -- and expectancy near zero. Measured on 100,000 M5 bars each:

  symbol            win%    expR      z vs 25%   verdict
  Step Index        26.2   +0.046       +0.32    coin flip
  Step Index 500    25.0   -0.000       +0.00    coin flip
  Step Index 200    23.1   -0.077       -0.47    coin flip
  Multi Step 2      31.3   +0.252       +1.67    coin flip
  XAUUSD            40.9   +0.628       +3.94    real edge

    python scripts/fetch_mt5_data.py --symbol "Step Index" --out data/Step_Index_M5.csv
    python scripts/null_hypothesis_test.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import COMBINATION_F, SYMBOL_SPECS, BrokerSpec, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402

from experiments import run, walk_forward_score  # noqa: E402

#: Synthetic specifications read from the live terminal.
SYNTHETIC_SPECS = {
    "Step Index": BrokerSpec(symbol="Step Index", contract_size=10.0, tick_size=0.1,
                             tick_value=1.0, volume_step=0.01, volume_min=0.1,
                             volume_max=100.0, default_spread_points=2.0),
    "Step Index 200": BrokerSpec(symbol="Step Index 200", contract_size=10.0, tick_size=0.1,
                                 tick_value=1.0, volume_step=0.01, volume_min=0.1,
                                 volume_max=50.0, default_spread_points=2.0),
    "Step Index 500": BrokerSpec(symbol="Step Index 500", contract_size=10.0, tick_size=0.1,
                                 tick_value=1.0, volume_step=0.01, volume_min=0.1,
                                 volume_max=50.0, default_spread_points=5.0),
    "Multi Step 2 Index": BrokerSpec(symbol="Multi Step 2 Index", contract_size=10.0,
                                     tick_size=0.01, tick_value=0.01, volume_step=0.01,
                                     volume_min=0.1, volume_max=50.0, default_spread_points=25.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reference", default="data/XAUUSD_M5_live.csv",
                        help="the real instrument, for contrast")
    parser.add_argument("--risk", type=float, default=0.5,
                        help="low enough that the drawdown lock truncates nothing")
    parser.add_argument("--balance", type=float, default=10000.0)
    args = parser.parse_args()

    cases = []
    for name, spec in SYNTHETIC_SPECS.items():
        path = os.path.join(args.data_dir, name.replace(" ", "_") + "_M5.csv")
        if os.path.exists(path):
            cases.append((name, path, spec))
    if os.path.exists(args.reference):
        cases.append(("XAUUSD (real)", args.reference, SYMBOL_SPECS["XAUUSD"]))
    if not cases:
        print("No data. Fetch a synthetic first, e.g.:")
        print('  python scripts/fetch_mt5_data.py --symbol "Step Index" --out data/Step_Index_M5.csv')
        return

    break_even = 100.0 / (1.0 + StrategyConfig().reward_risk)
    rows = []
    for name, path, spec in cases:
        raw = load_m5_csv(path)
        cfg = replace(StrategyConfig(broker=spec, regime_timeframes=("H4", "H1", "M30", "M5"),
                                     risk_percent_initial_balance=args.risk, **COMBINATION_F))
        res = run(raw, cfg, args.balance)
        m = res.metrics
        wf = walk_forward_score(raw, cfg, args.balance, 30000, 14000)
        n, win = m["trades"], m["win_rate"] / 100.0
        p0 = break_even / 100.0
        # Against the payoff's geometric break-even, not against zero.
        z = (win - p0) / math.sqrt(p0 * (1 - p0) / n) if n else 0.0
        rows.append({
            "symbol": name, "trades": n,
            "win_%": round(m["win_rate"], 1),
            "exp_R": round(m["expectancy_r"], 3),
            "PF": round(m["profit_factor"], 2) if m["profit_factor"] == m["profit_factor"] else None,
            "maxDD_%": round(m["max_drawdown_percent"], 2),
            "z_vs_breakeven": round(z, 2),
            "verdict": "REAL EDGE" if z > 1.96 else ("below" if z < -1.96 else "coin flip"),
            "wf": f"{wf['profitable']}/{wf['windows']}",
        })
        print(f"  ran {name}", flush=True)

    print(f"\n=== Null hypothesis test (break-even for 1:{StrategyConfig().reward_risk:g} "
          f"is {break_even:.1f}%) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  A synthetic scoring REAL EDGE means the configuration is fitting noise:")
    print("  there is nothing in a constructed random walk for it to have found.")


if __name__ == "__main__":
    main()
