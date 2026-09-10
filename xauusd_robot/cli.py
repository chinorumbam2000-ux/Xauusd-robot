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
from .config import StrategyConfig
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
    config = replace(StrategyConfig(), risk_percent_initial_balance=args.risk)
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
    table = risk_matrix_comparison(m5, StrategyConfig(), args.balance)
    print("\n=== Risk research matrix (Section 5.1) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nChoose a production risk from evidence (drawdown + robustness), never from intuition.")


def cmd_sweep(args) -> None:
    m5 = load_m5_csv(args.data)
    values = _parse_values(args.values)
    table = parameter_sweep(m5, args.param, values, StrategyConfig(), args.balance)
    print(f"\n=== Parameter sweep: {args.param} (Section 10) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nPrefer a broad plateau across neighbouring values over a single narrow peak.")


def cmd_spread_stress(args) -> None:
    m5 = load_m5_csv(args.data)
    table = spread_stress_test(m5, base_config=StrategyConfig(), initial_balance=args.balance)
    print("\n=== Spread / execution stress test ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def cmd_split(args) -> None:
    m5 = load_m5_csv(args.data)
    config = replace(StrategyConfig(), risk_percent_initial_balance=args.risk)
    for name, segment in chronological_split(m5).items():
        if len(segment) < config.ema_period * 2:
            print(f"\n{name}: too few bars ({len(segment)}) for a meaningful run")
            continue
        result = Backtester(segment, config, args.balance).run()
        _print_metrics(result.metrics, f"{name} (risk={args.risk}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xauusd_robot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True, help="path to M5 OHLC CSV")
    common.add_argument("--balance", type=float, default=1000.0, help="initial account balance")

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

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
