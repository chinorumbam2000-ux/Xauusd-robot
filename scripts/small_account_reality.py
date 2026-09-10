"""What a $100 XAUUSD account can and cannot do, using real broker constraints.

Two hard limits collide on a small gold account:

1. Minimum lot size. Deriv's XAUUSD minimum is 0.01 lots = 1 ounce, so a $1
   move is $1. A structural stop of $5 therefore risks $5 -- 5% of a $100
   account -- and there is no smaller position available. The account cannot
   choose to risk less.

2. Compounding required. Turning $100 into $2,000 in a month is 20x. At five
   trades a day that is ~105 trades, needing ~2.9% growth per trade, every
   trade, without a bad run.

This script measures the outcome distribution by resampling the R-multiples
actually produced by the strategy on real data, rather than assuming any.

    python scripts/small_account_reality.py --data data/XAUUSD_M5_live.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import DERIV_XAUUSD, BrokerSpec, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402

from experiments import run  # noqa: E402

COMBO_F = {
    "ob_require_swing_break": False, "wpr_max_lead_bars": 5, "setup_expiry_bars": 8,
    "sr_zone_atr": 0.15, "push_body_ratio": 0.65,
}


def simulate(r_values, balance, risk_fraction, n_trades, sims, target, ruin_frac, seed=11):
    """Resample real R outcomes; report where the account ends up.

    Risk is a fixed fraction of the CURRENT balance here (aggressive
    compounding), which is the only way a small account could grow quickly --
    and the reason it can also collapse quickly.
    """
    rng = np.random.default_rng(seed)
    hit_target = lockout = ruined = 0
    finals = []
    for _ in range(sims):
        equity, peak = balance, balance
        for _ in range(n_trades):
            equity += rng.choice(r_values) * equity * risk_fraction
            peak = max(peak, equity)
            if equity <= balance * ruin_frac:
                ruined += 1
                break
            if (1 - equity / peak) >= 0.15:
                lockout += 1
                break
            if equity >= target:
                hit_target += 1
                break
        finals.append(max(equity, 0.0))
    finals = np.array(finals)
    return {
        "risk_%": round(risk_fraction * 100, 1),
        "P(reach target)": f"{hit_target / sims:.1%}",
        "P(15% lockout)": f"{lockout / sims:.1%}",
        "P(down 50%+)": f"{ruined / sims:.1%}",
        "median_final": round(float(np.median(finals)), 0),
        "p10": round(float(np.percentile(finals, 10)), 0),
        "p90": round(float(np.percentile(finals, 90)), 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--target", type=float, default=2000.0)
    parser.add_argument("--trades", type=int, default=105, help="5/day x 21 trading days")
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()

    m5 = load_m5_csv(args.data)
    config = replace(
        StrategyConfig(broker=DERIV_XAUUSD), regime_timeframes=("M15", "M5"),
        risk_percent_initial_balance=1.0, **COMBO_F,
    )
    result = run(m5, config, 10000.0)
    trades = result.trades
    r_values = trades["r_multiple"].to_numpy()

    print("=== Strategy's real outcome distribution (best tested config) ===")
    print(f"  trades         : {len(r_values)}")
    print(f"  win rate       : {(r_values > 0).mean():.1%}")
    print(f"  expectancy     : {r_values.mean():+.3f}R")

    stops = trades["stop_distance"]
    broker = DERIV_XAUUSD
    # 0.01 lots on XAUUSD = 1 oz, so risk in dollars == stop distance in dollars.
    min_risk = stops * broker.volume_min * broker.contract_size
    print("\n=== The minimum-lot floor on a small account ===")
    print(f"  typical stop distance : ${stops.median():.2f}  (p10 ${stops.quantile(.1):.2f}, p90 ${stops.quantile(.9):.2f})")
    print(f"  risk at 0.01 lots     : ${min_risk.median():.2f} per trade -- the smallest position that exists")
    for bal in (100, 500, 1000, 5000):
        pct = min_risk.median() / bal * 100
        verdict = "IMPOSSIBLE below this" if pct > 5 else "workable" if pct <= 2 else "aggressive"
        print(f"    on a ${bal:>5,} account that is {pct:5.1f}% per trade  <- {verdict}")

    print(f"\n=== ${args.balance:,.0f} -> ${args.target:,.0f} in {args.trades} trades ===")
    growth = (args.target / args.balance) ** (1 / args.trades) - 1
    print(f"  required compounding: {growth:.2%} per trade, every trade, for {args.trades} trades")
    print(f"  at {r_values.mean():+.3f}R expectancy that needs ~{growth / r_values.mean() * 100:.1f}% risk per trade\n")

    rows = [
        simulate(r_values, args.balance, risk, args.trades, args.sims, args.target, 0.5)
        for risk in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nP(reach target) is the 'flip the account' outcome; P(down 50%+) is the")
    print("same strategy on a different draw. They are produced by the same settings.")


if __name__ == "__main__":
    main()
