"""Compare which timeframes belong in the EMA200 regime filter.

Section 3.1 specifies six. Fewer timeframes align more often, so the system
trades more -- but the filter is also the only thing establishing trend
direction, and every timeframe removed is confirmation given up.

Two confounds are controlled here:

* Warm-up. The slowest timeframe sets it (D1 ~57,600 M5 bars, H4 ~14,400,
  H1 ~2,400). A faster set leaves more of the file tradeable, so raw trade
  counts flatter it. trades_per_month is the comparable figure.
* Aggregate return on one sample says little, so every variant is
  walk-forward tested and consistency is reported beside it.

    python scripts/regime_timeframes.py --data data/XAUUSD_M5_live.csv
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

from experiments import run, walk_forward_score  # noqa: E402

VARIANTS = {
    "D1,H4,H1,M30,M15,M5 (blueprint)": ("D1", "H4", "H1", "M30", "M15", "M5"),
    "H4,H1,M30,M15,M5 (current)": ("H4", "H1", "M30", "M15", "M5"),
    "H4,H1,M15,M5": ("H4", "H1", "M15", "M5"),
    "H1,M30,M15,M5": ("H1", "M30", "M15", "M5"),
    "H1,M15,M5": ("H1", "M15", "M5"),
    "H1,M15 (requested)": ("H1", "M15"),
    "H1,M5": ("H1", "M5"),
    "M15,M5": ("M15", "M5"),
    "H1 only": ("H1",),
}

#: The best combination from scripts/combinations.py, to check whether the
#: timeframe conclusion holds under a different rule set rather than only
#: under the defaults.
COMBO_F = {
    "ob_require_swing_break": False,
    "wpr_max_lead_bars": 5,
    "setup_expiry_bars": 8,
    "sr_zone_atr": 0.15,
    "push_body_ratio": 0.65,
}


def evaluate(m5, config, balance, train_bars, test_bars, label) -> dict:
    result = run(m5, config, balance)
    m = result.metrics
    wf = walk_forward_score(m5, config, balance, train_bars, test_bars)
    return {
        "regime": label,
        "n_tf": len(config.regime_timeframes),
        "trades": m["trades"],
        "per_month": round(m["trades_per_month"], 2),
        "win_%": round(m["win_rate"], 1),
        "exp_R": round(m["expectancy_r"], 3),
        "PF": round(m["profit_factor"], 2) if m["profit_factor"] == m["profit_factor"] else None,
        "net_%": round(m["net_return_percent"], 1),
        "maxDD_%": round(m["max_drawdown_percent"], 2),
        "streak": m["max_consecutive_losses"],
        "wf_win": f"{wf['profitable']}/{wf['windows']}",
        "wf_R": wf["wf_total_r"],
        "wf_worst": wf["worst_r"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--broker", choices=("default", "deriv"), default="deriv")
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--test-bars", type=int, default=14000)
    parser.add_argument("--with-combo", action="store_true", help="also run every variant under combination F")
    args = parser.parse_args()

    m5 = load_m5_csv(args.data)
    broker = DERIV_XAUUSD if args.broker == "deriv" else BrokerSpec()
    base = replace(StrategyConfig(broker=broker), risk_percent_initial_balance=args.risk)

    print(f"data: {len(m5):,} bars  {m5.index[0].date()} -> {m5.index[-1].date()}   risk {args.risk}%\n")

    rows = []
    for label, timeframes in VARIANTS.items():
        config = replace(base, regime_timeframes=timeframes)
        rows.append(evaluate(m5, config, args.balance, args.train_bars, args.test_bars, label))
        print(f"  ran {label}")
    print("\n=== Regime timeframe comparison (baseline rules) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    if args.with_combo:
        combo_rows = []
        for label, timeframes in VARIANTS.items():
            config = replace(base, regime_timeframes=timeframes, **COMBO_F)
            combo_rows.append(evaluate(m5, config, args.balance, args.train_bars, args.test_bars, label))
            print(f"  ran {label} (combo F)")
        print("\n=== Regime timeframe comparison (combination F rules) ===")
        print(pd.DataFrame(combo_rows).to_string(index=False))

    print("\nper_month is the fair trade-frequency column: a faster timeframe set")
    print("warms up sooner and leaves more of the file tradeable, which flatters")
    print("raw trade counts. Judge robustness on wf_win and wf_worst.")


if __name__ == "__main__":
    main()
