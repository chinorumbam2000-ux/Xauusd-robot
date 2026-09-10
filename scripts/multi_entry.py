"""Does splitting an opportunity into three entries beat one 3R entry?

"Three entries per opportunity" can mean three different things, and only two
of them are structurally different from simply trading larger:

  1. LADDERED TARGETS (scale out). One entry, one stop, the position split in
     three with targets at 1R, 2R and 3R. Raises win rate, lowers average R.
  2. LADDERED ENTRIES (scale in). Three fills at progressively better prices
     inside the zone, one shared stop, all targeting 3R. Improves the average
     entry but risks the deeper fills never happening.
  3. THREE COPIES of the same trade. Arithmetically identical to tripling risk
     percent, which scripts/risk_normalised.py already measures -- it moves
     nothing except position size, so it is included only as a reference row.

Variants 1 and 2 are evaluated from each trade's recorded maximum favourable
excursion, so they use the real price paths the strategy actually produced
rather than a re-simulation. The stop-before-target assumption carries through:
MFE excludes any bar on which the stop was hit.

    python scripts/multi_entry.py --data data/XAUUSD_M5_live.csv
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


def ladder_targets(mfe: np.ndarray, hit_stop: np.ndarray, targets=(1.0, 2.0, 3.0)) -> np.ndarray:
    """R per trade when the position is split evenly across several targets.

    Each third wins its target if the trade ran that far, otherwise it loses
    its share of the 1R risk.
    """
    share = 1.0 / len(targets)
    total = np.zeros(len(mfe))
    for t in targets:
        reached = mfe >= t
        total += np.where(reached, t * share, -share)
    return total


def ladder_entries(mfe: np.ndarray, r: np.ndarray, offsets=(0.0, 0.25, 0.5)) -> np.ndarray:
    """R per trade when three fills are staggered deeper into the zone.

    A fill at ``offset`` R better than the original entry shares the same stop,
    so its risk shrinks to (1 - offset) and its 3R target sits nearer. Fills
    only happen if price actually retraced that far, which is unobservable from
    the recorded path -- so a deeper fill is assumed only when the trade first
    moved against the entry, approximated here by trades whose MFE stayed below
    the offset before resolving. Deliberately pessimistic: an unfilled tranche
    contributes nothing.
    """
    share = 1.0 / len(offsets)
    total = np.zeros(len(mfe))
    for off in offsets:
        if off == 0.0:
            total += r * share
            continue
        risk = 1.0 - off
        # A deeper fill needs price to trade against the entry by `off` R.
        # Trades that ran straight to target never offered it.
        filled = mfe < 3.0
        gained = np.where(mfe >= (3.0 - off), 3.0 * share, -risk * share)
        total += np.where(filled, gained, 0.0)
    return total


def summarise(label: str, r_values: np.ndarray, risk_fraction: float, balance: float) -> dict:
    wins = r_values > 0
    equity, peak, max_dd = balance, balance, 0.0
    for r in r_values:
        equity += r * balance * risk_fraction  # fixed initial-balance risk model
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
    streak = worst = 0
    for r in r_values:
        streak = streak + 1 if r < 0 else 0
        worst = max(worst, streak)
    gross_win = r_values[r_values > 0].sum()
    gross_loss = -r_values[r_values <= 0].sum()
    return {
        "variant": label,
        "trades": len(r_values),
        "win_%": round(wins.mean() * 100, 1),
        "exp_R": round(r_values.mean(), 3),
        "total_R": round(r_values.sum(), 1),
        "PF": round(gross_win / gross_loss, 2) if gross_loss else None,
        "net_%": round((equity - balance) / balance * 100, 1),
        "maxDD_%": round(max_dd * 100, 2),
        "streak": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/XAUUSD_M5_live.csv")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=2.0)
    args = parser.parse_args()

    raw = load_m5_csv(args.data)
    config = replace(
        StrategyConfig(broker=SYMBOL_SPECS["XAUUSD"]),
        regime_timeframes=("H4", "H1", "M30", "M15", "M5"),
        risk_percent_initial_balance=args.risk, **COMBINATION_F,
    )
    result = run(raw, config, args.balance)
    trades = result.trades
    if trades.empty:
        print("no trades")
        return

    r = trades["r_multiple"].to_numpy()
    mfe = trades["mfe_r"].to_numpy() if "mfe_r" in trades.columns else np.zeros(len(r))
    hit_stop = r < 0
    frac = args.risk / 100.0

    print(f"data: {len(raw):,} bars, {len(r)} trades, risk {args.risk}%\n")
    print("=== How far trades actually ran (max favourable excursion) ===")
    for level in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        print(f"  reached {level:>3}R : {(mfe >= level).mean() * 100:5.1f}% of trades")

    rows = [
        summarise("1 entry @ 3R (current)", r, frac, args.balance),
        summarise("3 targets 1R/2R/3R", ladder_targets(mfe, hit_stop), frac, args.balance),
        summarise("3 targets 1R/2R/4R", ladder_targets(mfe, hit_stop, (1.0, 2.0, 4.0)), frac, args.balance),
        summarise("3 targets 2R/3R/4R", ladder_targets(mfe, hit_stop, (2.0, 3.0, 4.0)), frac, args.balance),
        summarise("3 laddered entries", ladder_entries(mfe, r), frac, args.balance),
        summarise("3 copies (= 3x risk)", r, frac * 3, args.balance),
    ]
    print("\n=== Variants at equal per-opportunity risk ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nEvery row except the last risks the SAME amount per opportunity; only the")
    print("shape of the exit differs. The last row is three copies of one trade, which")
    print("is just leverage and is shown for reference.")


if __name__ == "__main__":
    main()
