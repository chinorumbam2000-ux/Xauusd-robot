"""Compare configurations at equal drawdown risk, not equal risk percent.

Comparing net return at a fixed 1% risk flatters loose configurations: they
earn more because they take more risk, not because they are better. The
constraint that actually binds is the 15% peak-equity lockout (Section 5.5),
which is a hard stop requiring manual review.

So each configuration is swept across risk percentages, and the comparable
figure is the best net return it can reach while keeping max drawdown inside
a stated budget. That is what the account can actually run.

    python scripts/risk_normalised.py --data data/XAUUSD_M5_live.csv
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

from experiments import run  # noqa: E402

COMBO_F = {
    "ob_require_swing_break": False,
    "wpr_max_lead_bars": 5,
    "setup_expiry_bars": 8,
    "sr_zone_atr": 0.15,
    "push_body_ratio": 0.65,
}

CONFIGS = {
    "H4,H1,M30,M15,M5 + F": ("H4", "H1", "M30", "M15", "M5"),
    "H1,M30,M15,M5 + F": ("H1", "M30", "M15", "M5"),
    "H1,M15,M5 + F": ("H1", "M15", "M5"),
    "M15,M5 + F": ("M15", "M5"),
    "H1,M15 + F": ("H1", "M15"),
    "H1 only + F": ("H1",),
}

RISKS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--broker", choices=("default", "deriv"), default="deriv")
    parser.add_argument("--dd-budget", type=float, default=12.0,
                        help="max drawdown %% allowed, kept under the 15%% lockout")
    args = parser.parse_args()

    m5 = load_m5_csv(args.data)
    broker = DERIV_XAUUSD if args.broker == "deriv" else BrokerSpec()

    detail, best = [], []
    for label, timeframes in CONFIGS.items():
        rows = []
        for risk in RISKS:
            config = replace(
                StrategyConfig(broker=broker), regime_timeframes=timeframes,
                risk_percent_initial_balance=risk, **COMBO_F,
            )
            m = run(m5, config, args.balance).metrics
            row = {
                "config": label, "risk_%": risk, "trades": m["trades"],
                "net_%": round(m["net_return_percent"], 1),
                "maxDD_%": round(m["max_drawdown_percent"], 2),
                "final": round(m["final_balance"], 0),
            }
            rows.append(row)
            detail.append(row)
        print(f"  swept {label}")

        within = [r for r in rows if r["maxDD_%"] <= args.dd_budget]
        if within:
            top = max(within, key=lambda r: r["net_%"])
            best.append({
                "config": label, "max_risk_%": top["risk_%"], "trades": top["trades"],
                "net_%": top["net_%"], "maxDD_%": top["maxDD_%"], "final_balance": top["final"],
            })
        else:
            best.append({"config": label, "max_risk_%": None, "trades": rows[0]["trades"],
                         "net_%": None, "maxDD_%": rows[0]["maxDD_%"], "final_balance": None})

    print("\n=== Full sweep ===")
    print(pd.DataFrame(detail).to_string(index=False))

    print(f"\n=== Best achievable within a {args.dd_budget:.0f}% drawdown budget ===")
    table = pd.DataFrame(best).sort_values("net_%", ascending=False, na_position="last")
    print(table.to_string(index=False))
    print("\nThis is the honest comparison: what each configuration returns when held")
    print("to the same risk of tripping the 15% peak-equity lockout.")


if __name__ == "__main__":
    main()
