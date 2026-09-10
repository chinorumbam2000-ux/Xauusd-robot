"""Command line entry point for the XAUUSD research layer.

    python -m xauusd_robot.cli backtest --data data/XAUUSD_M5.csv --risk 1.0
    python -m xauusd_robot.cli risk-matrix --data data/XAUUSD_M5.csv
    python -m xauusd_robot.cli sweep --data data/XAUUSD_M5.csv --param setup_expiry_bars --values 3,5,8,10
    python -m xauusd_robot.cli spread-stress --data data/XAUUSD_M5.csv
    python -m xauusd_robot.cli split --data data/XAUUSD_M5.csv --risk 1.0
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import pandas as pd

from .backtester import Backtester
from .config import DERIV_XAUUSD, BrokerSpec, StrategyConfig

BROKERS = {"default": BrokerSpec(), "deriv": DERIV_XAUUSD}


def _config(risk: float | None = None, broker: str = "default") -> StrategyConfig:
    config = StrategyConfig(broker=BROKERS[broker])
    if risk is not None:
        config = replace(config, risk_percent_initial_balance=risk)
    return config
from .data import load_m5_csv
from .metrics import monte_carlo_target_probability
from .validation import (
    chronological_split,
    parameter_sweep,
    risk_matrix_comparison,
    spread_stress_test,
)

HEADLINE_KEYS = (
    "trades", "win_rate", "break_even_win_rate", "expectancy_r", "total_r",
    "profit_factor", "net_return_percent", "max_drawdown_percent",
    "max_consecutive_losses", "trades_per_month", "final_balance", "target_reached",
)


def _print_metrics(metrics: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    for key in HEADLINE_KEYS:
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"{key:<26} {value:,.4f}")
        else:
            print(f"{key:<26} {value}")
    for group in ("by_direction", "by_zone_type", "by_confluence", "by_session"):
        data = metrics.get(group)
        if data:
            print(f"\n-- {group} --")
            print(pd.DataFrame(data).T.to_string(float_format=lambda v: f"{v:,.3f}"))

    funnel = metrics.get("funnel")
    if funnel:
        print("\n-- setup funnel (why the robot did or did not act) --")
        order = [
            "bars", "regime_BUY", "regime_SELL", "regime_MIXED",
            "zones_created_OB", "zones_created_FVG", "zones_created_SR",
            "zone_reactions", "zone_reactions_regime_matched", "setups_armed",
            "setups_reached_wpr_extreme", "setups_reached_wpr_confirmed",
            "setups_reached_push1", "setups_reached_push2",
            "setups_entry_evaluated", "orders_placed",
        ]
        for key in order:
            if key in funnel:
                print(f"{key:<32} {funnel[key]:>10,}")
        for key in sorted(k for k in funnel if k.startswith(("rejected_", "setup_expired", "setup_zone"))):
            print(f"{key:<32} {funnel[key]:>10,}")


def _parse_values(text: str):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part) if part.isdigit() or (part.startswith("-") and part[1:].isdigit()) else float(part))
        except ValueError:
            values.append(part)
    return values


def cmd_backtest(args) -> None:
    m5 = load_m5_csv(args.data)
    config = _config(args.risk, args.broker)
    result = Backtester(m5, config, args.balance).run()
    _print_metrics(result.metrics, f"Backtest risk={args.risk}%")

    if not result.trades.empty:
        mc = monte_carlo_target_probability(
            result.trades["r_multiple"].to_numpy(),
            initial_balance=args.balance,
            risk_money=args.balance * config.risk_fraction(),
            target_multiplier=config.target_multiplier,
            max_drawdown_lock=config.max_peak_equity_drawdown,
        )
        print("\n-- monte carlo (resampled sequence risk) --")
        for k, v in mc.items():
            print(f"{k:<26} {v:.3f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        result.trades.to_csv(f"{args.out}/trades.csv", index=False)
        result.events.to_csv(f"{args.out}/events.csv", index=False)
        result.equity.to_csv(f"{args.out}/equity.csv", index=False)
        with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as fh:
            json.dump(result.metrics, fh, indent=2, default=str)
        print(f"\nlogs written to {args.out}")
        if args.chart:
            from .reporting import plot_run

            path = plot_run(result, args.out)
            if path:
                print(f"chart written to {path}")


def cmd_risk_matrix(args) -> None:
    m5 = load_m5_csv(args.data)
    table = risk_matrix_comparison(m5, _config(broker=args.broker), args.balance)
    print("\n=== Risk research matrix (Section 5.1) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nChoose a production risk from evidence (drawdown + robustness), never from intuition.")


def cmd_sweep(args) -> None:
    m5 = load_m5_csv(args.data)
    values = _parse_values(args.values)
    table = parameter_sweep(m5, args.param, values, _config(broker=args.broker), args.balance)
    print(f"\n=== Parameter sweep: {args.param} (Section 10) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nPrefer a broad plateau across neighbouring values over a single narrow peak.")


def cmd_spread_stress(args) -> None:
    m5 = load_m5_csv(args.data)
    table = spread_stress_test(m5, base_config=_config(broker=args.broker), initial_balance=args.balance)
    print("\n=== Spread / execution stress test ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def cmd_split(args) -> None:
    m5 = load_m5_csv(args.data)
    config = _config(args.risk, args.broker)
    for name, segment in chronological_split(m5).items():
        if len(segment) < config.ema_period * 2:
            print(f"\n{name}: too few bars ({len(segment)}) for a meaningful run")
            continue
        result = Backtester(segment, config, args.balance).run()
        _print_metrics(result.metrics, f"{name} (risk={args.risk}%)")


def cmd_walk_forward(args) -> None:
    """Section 9.2: repeat development/validation across rolling chronological windows."""
    from .validation import run_backtest, walk_forward_windows

    m5 = load_m5_csv(args.data)
    config = _config(args.risk, args.broker)
    windows = walk_forward_windows(m5, args.train_bars, args.test_bars)
    if not windows:
        print(f"\nNot enough data for walk-forward: {len(m5):,} bars, need at least "
              f"{args.train_bars + args.test_bars:,}.")
        print("Remember the D1 EMA200 alone consumes ~57,600 bars of warm-up.")
        return

    from .metrics import compute_metrics

    rows = []
    for n, window in enumerate(windows, 1):
        # The test window must inherit indicator warm-up from its training
        # history -- a D1 EMA200 needs ~57,600 M5 bars before it means
        # anything, so scoring a bare test slice would score noise. Run over
        # train+test, then count only trades that ENTER inside the test window.
        combined = pd.concat([window["train"], window["test"]])
        test_start = window["test"].index[0]
        result = run_backtest(combined, config, args.balance)

        trades = result.trades
        if not trades.empty:
            trades = trades[trades["entry_time"] >= test_start].copy()
        m = compute_metrics(trades, result.equity.iloc[len(window["train"]):],
                            args.balance, config)
        rows.append({
            "window": n,
            "test_start": test_start.date(),
            "test_end": window["test"].index[-1].date(),
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "expectancy_r": m["expectancy_r"],
            "total_r": m["total_r"],
            "max_drawdown_percent": m["max_drawdown_percent"],
        })

    table = pd.DataFrame(rows)
    print(f"\n=== Walk-forward: {len(windows)} windows "
          f"({args.train_bars:,} train / {args.test_bars:,} test bars) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    profitable = int((table["total_r"] > 0).sum())
    print(f"\nprofitable windows : {profitable}/{len(table)}")
    print(f"total trades       : {int(table['trades'].sum())}")
    print("Consistency across windows matters more than the sum; a single dominant "
          "window is not persistence.")


def cmd_learning(args) -> None:
    """Report what live results say so far, and what they cannot yet say."""
    from .adaptive import DEFAULT_EXPECTATION, PerformanceTracker, TradeLedger

    ledger = TradeLedger(args.ledger).load()
    tracker = PerformanceTracker(ledger, DEFAULT_EXPECTATION)
    report = tracker.assess()

    print(f"\n=== Live learning status ({args.ledger}) ===")
    for key in ("verdict", "live_trades", "live_win_rate", "win_rate_ci_95",
                "live_expectancy_r", "live_total_r", "expected_win_rate",
                "expected_expectancy_r", "break_even_win_rate"):
        if key in report:
            print(f"  {key:<24} {report[key]}")
    print(f"\n  {report['detail']}")

    print("\n=== How much data buys how much certainty ===")
    print(pd.DataFrame(tracker.precision_table()).to_string(index=False))

    print(f"\n  Auto-tuning: {report['auto_tuning']}")
    if report["trades_until_refit"]:
        print(f"  {report['trades_until_refit']} more trades before `reoptimise` "
              f"can say anything defensible.")
    else:
        print("  Sample is large enough to re-run the sweep including live trades.")


def cmd_reoptimise(args) -> None:
    """Re-run the parameter sweep including live results -- report only.

    Nothing is applied automatically. Section 10's selection rule is a broad
    plateau across neighbouring values, which a human has to look at.
    """
    from .adaptive import DEFAULT_EXPECTATION, PerformanceTracker, TradeLedger

    ledger = TradeLedger(args.ledger).load()
    tracker = PerformanceTracker(ledger, DEFAULT_EXPECTATION)
    report = tracker.assess()
    n = report["live_trades"]

    print(f"\nlive trades in ledger: {n}")
    if n < tracker.MIN_TRADES_FOR_REFIT and not args.force:
        print(f"\nREFUSING to re-optimise on {n} trades.")
        print(f"  {tracker.MIN_TRADES_FOR_REFIT} is the minimum for a defensible re-fit; at "
              f"~5 trades a month that is {(tracker.MIN_TRADES_FOR_REFIT - n) / 5:.0f} more months.")
        print("  Re-fitting on a smaller sample fits noise, and would do so repeatedly.")
        print("  Pass --force to override, understanding that.")
        return

    print("\nLive results:")
    print(f"  win rate {report.get('live_win_rate')}% "
          f"(95% CI {report.get('win_rate_ci_95')}), expectancy "
          f"{report.get('live_expectancy_r')}R")
    print(f"  verdict: {report['verdict']} -- {report['detail']}")
    print("\nRun scripts/experiments.py and scripts/combinations.py against a data file")
    print("that now includes the live period, then compare against the current config.")
    print("Apply a change only if it holds across walk-forward windows, not because")
    print("it raises the aggregate.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xauusd_robot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True, help="path to M5 OHLC CSV")
    common.add_argument("--balance", type=float, default=1000.0, help="initial account balance")
    common.add_argument("--broker", choices=sorted(BROKERS), default="default",
                        help="broker symbol specification (deriv = live Deriv-Demo XAUUSD spec)")

    p = sub.add_parser("backtest", parents=[common])
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--out", default=None, help="directory for trade/event/equity logs")
    p.add_argument("--chart", action="store_true", help="also write a diagnostic PNG (needs matplotlib)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("risk-matrix", parents=[common])
    p.set_defaults(func=cmd_risk_matrix)

    p = sub.add_parser("sweep", parents=[common])
    p.add_argument("--param", required=True)
    p.add_argument("--values", required=True, help="comma separated values")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("spread-stress", parents=[common])
    p.set_defaults(func=cmd_spread_stress)

    p = sub.add_parser("split", parents=[common])
    p.add_argument("--risk", type=float, default=1.0)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("walk-forward", parents=[common])
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--train-bars", type=int, default=80000)
    p.add_argument("--test-bars", type=int, default=20000)
    p.set_defaults(func=cmd_walk_forward)

    # These two read the live ledger and need no market data file.
    p = sub.add_parser("learning")
    p.add_argument("--ledger", default="state/trade_ledger.json")
    p.set_defaults(func=cmd_learning)

    p = sub.add_parser("reoptimise")
    p.add_argument("--ledger", default="state/trade_ledger.json")
    p.add_argument("--force", action="store_true",
                   help="re-fit on an inadequate sample anyway")
    p.set_defaults(func=cmd_reoptimise)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
