"""5% per opportunity, split across three orders, on standard XAUUSD.

Two things decide whether this works, and they pull in opposite directions.

Lot granularity. Standard XAUUSD is a 100 oz contract with a 0.01 lot minimum,
so the smallest order that exists risks the stop distance x $100 x 0.01. At the
median structural stop that is $12.47. A tranche sized at 1.67% of the account
can only exist if 1.67% of the balance is at least that much, which puts a hard
floor under the account size before three orders are even placeable.

The 15% lockout. It is enforced here, unlike in a naive equity projection,
because at 5% risk it is the dominant outcome rather than an edge case. A
projection that ignores it reports returns the account would never be allowed
to earn.

Both the "three orders, same target" and "three orders, laddered targets"
readings are tested, because they are not the same thing: the former is one
position wearing three tickets, the latter changes the payoff.

    python scripts/split_risk_test.py
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

from experiments import run  # noqa: E402

LOCKOUT = 0.15


def tranche_lots(budget: float, stop: float, spec) -> float:
    """Broker-normalised lots for a risk budget, rounded DOWN as Section 5.3 requires."""
    per_lot = stop / spec.tick_size * spec.tick_value
    if per_lot <= 0:
        return 0.0
    raw = budget / per_lot
    lots = np.floor(raw / spec.volume_step) * spec.volume_step
    return max(0.0, round(lots, 2))


def simulate(r, mfe, stops, spec, balance, total_risk_pct, tranches, mode):
    """Walk the real trade sequence with real lot rounding and the lockout live."""
    equity, peak, max_dd = balance, balance, 0.0
    placed = skipped = 0
    locked_at = None
    share = total_risk_pct / 100.0 / tranches
    targets = {1: (3.0,), 3: (1.0, 2.0, 3.0)}[tranches] if mode == "ladder" else (3.0,) * tranches

    for k in range(len(r)):
        if locked_at is not None:
            break
        stop = stops[k]
        filled_any = False
        for t in targets:
            lots = tranche_lots(equity * share, stop, spec)
            if lots < spec.volume_min:
                skipped += 1
                continue
            risk_money = lots * stop / spec.tick_size * spec.tick_value
            # Outcome of this tranche: it wins its own target if the trade ran
            # that far, otherwise it takes the full stop.
            won = mfe[k] >= t
            equity += (t * risk_money) if won else (-risk_money)
            filled_any = True
        if filled_any:
            placed += 1
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0.0
        max_dd = max(max_dd, dd)
        if dd >= LOCKOUT:
            locked_at = placed

    return {
        "balance": balance,
        "opportunities_taken": placed,
        "tranches_too_small": skipped,
        "net_%": round((equity - balance) / balance * 100, 1),
        "final": round(equity, 0),
        "maxDD_%": round(max_dd * 100, 2),
        "locked_after": locked_at if locked_at is not None else "-",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/XAUUSD_M5_live.csv")
    parser.add_argument("--total-risk", type=float, default=5.0)
    args = parser.parse_args()

    spec = SYMBOL_SPECS["XAUUSD"]
    raw = load_m5_csv(args.data)
    cfg = replace(StrategyConfig(broker=spec), regime_timeframes=("H4", "H1", "M30", "M15", "M5"),
                  risk_percent_initial_balance=2.0, **COMBINATION_F)
    res = run(raw, cfg, 10000.0)
    tr = res.trades
    r = tr["r_multiple"].to_numpy()
    mfe = tr["mfe_r"].to_numpy()
    stops = tr["stop_distance"].to_numpy()

    print(f"{len(r)} real trades | median stop ${np.median(stops):.2f} "
          f"| min order risks ${np.median(stops) * spec.volume_min / spec.tick_size * spec.tick_value:.2f}\n")

    share = args.total_risk / 3
    floor_balance = (np.median(stops) * spec.volume_min / spec.tick_size * spec.tick_value) / (share / 100)
    print(f"=== Minimum account for three {share:.2f}% tranches on standard XAUUSD ===")
    print(f"  each tranche must risk at least the minimum order: "
          f"${np.median(stops) * spec.volume_min / spec.tick_size * spec.tick_value:.2f}")
    print(f"  so {share:.2f}% of the balance must reach that -> balance >= ${floor_balance:,.0f}\n")

    rows = []
    for bal in (100, 250, 500, 750, 1000, 2500, 5000, 10000):
        rows.append({"variant": f"1 order @ {args.total_risk}%",
                     **simulate(r, mfe, stops, spec, bal, args.total_risk, 1, "same")})
        rows.append({"variant": f"3 x {share:.2f}% same 3R target",
                     **simulate(r, mfe, stops, spec, bal, args.total_risk, 3, "same")})
        rows.append({"variant": f"3 x {share:.2f}% laddered 1R/2R/3R",
                     **simulate(r, mfe, stops, spec, bal, args.total_risk, 3, "ladder")})

    table = pd.DataFrame(rows)
    print(f"=== {args.total_risk}% per opportunity, standard XAUUSD, 15% lockout enforced ===")
    print(table.to_string(index=False))
    print("\n  tranches_too_small = orders skipped because 1/3 of the risk budget could")
    print("  not buy even 0.01 lots. locked_after = opportunities taken before the 15%")
    print("  peak-equity lockout disabled further entries.")


if __name__ == "__main__":
    main()
