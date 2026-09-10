"""Combine the single-parameter winners and check they survive together.

Changes that each look good alone often do not stack: they can loosen the
same bottleneck twice, or trade quality for quantity past the point where
the 1:3 payoff still clears its 25% break-even win rate. Every candidate is
therefore walk-forward tested, and consistency across windows is reported
next to the aggregate.

    python scripts/combinations.py --data data/XAUUSD_M5_live.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import DERIV_XAUUSD, BrokerSpec, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402

from experiments import run, summarise, walk_forward_score  # noqa: E402

#: Built from the single-parameter sweep, strongest first. The "quality" and
#: "kitchen sink" entries are included as controls: one tightens instead of
#: loosening, the other loosens everything at once so degradation is visible.
CANDIDATES = {
    "baseline": {},
    "A no-swing-break": {"ob_require_swing_break": False},
    "B A + wpr_lead=5": {"ob_require_swing_break": False, "wpr_max_lead_bars": 5},
    "C A + expiry=8": {"ob_require_swing_break": False, "setup_expiry_bars": 8},
    "D A + lead=5 + expiry=8": {
        "ob_require_swing_break": False, "wpr_max_lead_bars": 5, "setup_expiry_bars": 8,
    },
    "E D + sr_zone=0.15": {
        "ob_require_swing_break": False, "wpr_max_lead_bars": 5, "setup_expiry_bars": 8,
        "sr_zone_atr": 0.15,
    },
    "F E + push=0.65 (quality)": {
        "ob_require_swing_break": False, "wpr_max_lead_bars": 5, "setup_expiry_bars": 8,
        "sr_zone_atr": 0.15, "push_body_ratio": 0.65,
    },
    "G kitchen sink (control)": {
        "ob_require_swing_break": False, "wpr_max_lead_bars": 12, "setup_expiry_bars": 15,
        "sr_zone_atr": 0.15, "push_body_ratio": 0.50,
        "wpr_oversold": -65.0, "wpr_overbought": -35.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--broker", choices=("default", "deriv"), default="deriv")
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--test-bars", type=int, default=14000)
    args = parser.parse_args()

    m5 = load_m5_csv(args.data)
    broker = DERIV_XAUUSD if args.broker == "deriv" else BrokerSpec()
    base = replace(StrategyConfig(broker=broker), risk_percent_initial_balance=args.risk)

    print(f"data: {len(m5):,} bars  {m5.index[0].date()} -> {m5.index[-1].date()}")
    print(f"regime: {', '.join(base.regime_timeframes)}   risk {args.risk}%\n")

    rows = []
    for name, overrides in CANDIDATES.items():
        config = replace(base, **overrides) if overrides else base
        summary = summarise(name, "", run(m5, config, args.balance))
        wf = walk_forward_score(m5, config, args.balance, args.train_bars, args.test_bars)
        rows.append({
            "candidate": name,
            "trades": summary["trades"],
            "win_%": summary["win_rate"],
            "exp_R": summary["expectancy_r"],
            "PF": summary["profit_factor"],
            "net_%": summary["net_%"],
            "maxDD_%": summary["max_dd_%"],
            "streak": summary["max_losses"],
            "wf_win": f"{wf['profitable']}/{wf['windows']}",
            "wf_R": wf["wf_total_r"],
            "wf_worst": wf["worst_r"],
        })
        print(f"  ran {name}")

    table = pd.DataFrame(rows)
    print("\n=== Combination comparison ===")
    print(table.to_string(index=False))
    print("\nwf_win  = walk-forward windows profitable (consistency)")
    print("wf_worst= worst single window in R; negative means a losing stretch is normal")
    print("\nPrefer the candidate that raises trades while keeping wf_win at maximum and")
    print("wf_worst >= 0. Net % on the whole sample is the least trustworthy column here.")


if __name__ == "__main__":
    main()
