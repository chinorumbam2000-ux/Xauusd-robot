"""Section 10: Optimization Without Overfitting.

Sweeps one rule family at a time, then walk-forward validates the survivors.
The selection rule from the blueprint is deliberately not "highest return":

    Prefer a broad plateau of good performance across neighbouring parameter
    values. Avoid a narrow 'magic' setting that collapses when moved slightly;
    that is a common sign of overfitting.

So every sweep prints its whole surface, and the ranking step reports
walk-forward consistency (how many rolling windows were profitable) beside
the aggregate numbers. A configuration that wins on total return but only
profits in one window out of five is worse than it looks.

    python scripts/experiments.py --data data/XAUUSD_M5_live.csv --broker deriv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.backtester import Backtester  # noqa: E402
from xauusd_robot.config import DERIV_XAUUSD, BrokerSpec, StrategyConfig  # noqa: E402
from xauusd_robot.data import load_m5_csv  # noqa: E402
from xauusd_robot.metrics import compute_metrics  # noqa: E402
from xauusd_robot.validation import walk_forward_windows  # noqa: E402

#: One rule family per entry. Values bracket the baseline on both sides so a
#: plateau is visible rather than only the direction that loosens the rule.
SWEEPS = {
    "WPR extreme thresholds": [
        {"wpr_oversold": -80.0, "wpr_overbought": -20.0},
        {"wpr_oversold": -75.0, "wpr_overbought": -25.0},
        {"wpr_oversold": -70.0, "wpr_overbought": -30.0},
        {"wpr_oversold": -65.0, "wpr_overbought": -35.0},
        {"wpr_oversold": -60.0, "wpr_overbought": -40.0},
        {"wpr_oversold": -55.0, "wpr_overbought": -45.0},
    ],
    "WPR lead bars": [{"wpr_max_lead_bars": v} for v in (1, 2, 3, 5, 8, 12)],
    "Push body ratio": [{"push_body_ratio": v} for v in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)],
    "Setup expiry bars": [{"setup_expiry_bars": v} for v in (3, 5, 8, 10, 15, 20)],
    "Reaction window bars": [{"reaction_max_bars": v} for v in (1, 2, 3, 4, 6)],
    "Reward:risk": [{"reward_risk": v} for v in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)],
    "Max trades per day": [{"max_trades_per_day": v} for v in (2, 3, 5, 8, 20)],
    "Cooldown bars": [{"cooldown_bars": v} for v in (0, 1, 3, 6, 12)],
    "First reaction only": [{"first_reaction_only": v} for v in (True, False)],
    "Zone max age bars": [{"zone_max_age_bars": v} for v in (48, 96, 192, 288)],
    "FVG minimum size (ATR)": [{"fvg_min_atr": v} for v in (0.05, 0.10, 0.15, 0.20)],
    "S/R zone width (ATR)": [{"sr_zone_atr": v} for v in (0.05, 0.10, 0.15, 0.20)],
    "OB displacement (ATR)": [{"ob_displacement_atr": v} for v in (0.5, 0.75, 1.0, 1.25, 1.5)],
    "OB swing break required": [{"ob_require_swing_break": v} for v in (True, False)],
    "Max open positions": [{"max_open_positions": v} for v in (1, 2, 3)],
    "Stop buffer (ATR)": [{"sl_buffer_atr": v} for v in (0.05, 0.10, 0.20, 0.30)],
}


def run(m5, config, balance):
    return Backtester(m5, config, balance).run()


def summarise(label, value, result) -> dict:
    m = result.metrics
    funnel = m.get("funnel", {})
    return {
        "setting": label,
        "value": value,
        "trades": m["trades"],
        "win_rate": round(m["win_rate"], 1),
        "expectancy_r": round(m["expectancy_r"], 3),
        "total_r": round(m["total_r"], 1),
        "profit_factor": round(m["profit_factor"], 2) if m["profit_factor"] == m["profit_factor"] else None,
        "net_%": round(m["net_return_percent"], 2),
        "max_dd_%": round(m["max_drawdown_percent"], 2),
        "max_losses": m["max_consecutive_losses"],
        "armed": funnel.get("setups_armed", 0),
        "evaluated": funnel.get("setups_entry_evaluated", 0),
    }


def walk_forward_score(m5, config, balance, train_bars, test_bars) -> dict:
    """Consistency, not total return: how many rolling windows were profitable."""
    windows = walk_forward_windows(m5, train_bars, test_bars)
    if not windows:
        return {"windows": 0, "profitable": 0, "wf_trades": 0, "wf_total_r": 0.0, "worst_r": 0.0}

    total_r, trades, profitable, worst = 0.0, 0, 0, None
    for window in windows:
        combined = pd.concat([window["train"], window["test"]])
        start = window["test"].index[0]
        result = run(combined, config, balance)
        window_trades = result.trades
        if not window_trades.empty:
            window_trades = window_trades[window_trades["entry_time"] >= start]
        m = compute_metrics(window_trades, result.equity.iloc[len(window["train"]):], balance, config)
        total_r += m["total_r"]
        trades += m["trades"]
        profitable += 1 if m["total_r"] > 0 else 0
        worst = m["total_r"] if worst is None else min(worst, m["total_r"])
    return {
        "windows": len(windows),
        "profitable": profitable,
        "wf_trades": trades,
        "wf_total_r": round(total_r, 1),
        "worst_r": round(worst or 0.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--broker", choices=("default", "deriv"), default="deriv")
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--test-bars", type=int, default=14000)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    m5 = load_m5_csv(args.data)
    broker = DERIV_XAUUSD if args.broker == "deriv" else BrokerSpec()
    base = replace(StrategyConfig(broker=broker), risk_percent_initial_balance=args.risk)

    print(f"data: {len(m5):,} M5 bars  {m5.index[0]} -> {m5.index[-1]}")
    print(f"regime timeframes: {', '.join(base.regime_timeframes)}\n")

    baseline = summarise("BASELINE", "-", run(m5, base, args.balance))
    print("=== Baseline ===")
    print(pd.DataFrame([baseline]).to_string(index=False))

    all_rows = []
    for family, variants in SWEEPS.items():
        rows = []
        for overrides in variants:
            config = replace(base, **overrides)
            label = ", ".join(f"{k}={v}" for k, v in overrides.items())
            row = summarise(family, label, run(m5, config, args.balance))
            row["_overrides"] = overrides
            rows.append(row)
            all_rows.append(row)
        table = pd.DataFrame(rows).drop(columns=["setting", "_overrides"])
        print(f"\n=== {family} ===")
        print(table.to_string(index=False))

    # Rank by expectancy but require a trade count that is not a rounding error.
    candidates = [r for r in all_rows if r["trades"] >= max(20, baseline["trades"])]
    candidates.sort(key=lambda r: (r["total_r"], r["expectancy_r"]), reverse=True)
    shortlist, seen = [], set()
    for row in candidates:
        key = tuple(sorted(row["_overrides"].items()))
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(row)
        if len(shortlist) >= args.top:
            break

    print(f"\n\n=== Walk-forward check on the top {len(shortlist)} single changes ===")
    print("Aggregate return is not the test; surviving across windows is.\n")
    wf_rows = []
    base_wf = walk_forward_score(m5, base, args.balance, args.train_bars, args.test_bars)
    wf_rows.append({"setting": "BASELINE", "value": "-", **{k: baseline[k] for k in ("trades", "win_rate", "expectancy_r", "net_%", "max_dd_%")}, **base_wf})
    for row in shortlist:
        config = replace(base, **row["_overrides"])
        wf = walk_forward_score(m5, config, args.balance, args.train_bars, args.test_bars)
        wf_rows.append({
            "setting": row["setting"], "value": row["value"],
            **{k: row[k] for k in ("trades", "win_rate", "expectancy_r", "net_%", "max_dd_%")},
            **wf,
        })
    print(pd.DataFrame(wf_rows).to_string(index=False))

    print("\nRead this as: a change is only interesting if it raises trades AND keeps")
    print("'profitable' near 'windows' AND keeps 'worst_r' from going deeply negative.")
    print("A high net_% with 2/5 profitable windows is curve-fitting, not an edge.")


if __name__ == "__main__":
    main()
