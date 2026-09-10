"""Run the XAUUSD robot against a live MetaTrader 5 terminal (Blueprint Phase 6).

Dry run first -- prints what it would do, sends nothing:

    $env:MT5_LOGIN="..."; $env:MT5_SERVER="Broker-Demo"; $env:MT5_PASSWORD="..."
    python scripts/run_live.py --risk 1.0

Actually place orders on a DEMO account:

    python scripts/run_live.py --risk 1.0 --live

The trader refuses to run against a REAL-money account unless
--i-understand-this-is-real-money is passed. Do not pass it. The blueprint
requires out-of-sample, walk-forward and demo forward-testing gates before
real capital (Section 11.2), and this strategy has cleared none of them.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import StrategyConfig  # noqa: E402
from xauusd_robot.live import LiveTrader, LiveTraderError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--risk", type=float, default=1.0, help="percent of INITIAL balance per trade")
    parser.add_argument("--history-bars", type=int, default=60000,
                        help="warm-up window; the D1 EMA200 needs ~57,600 M5 bars")
    parser.add_argument("--terminal", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--state", default="state/live_state.json")
    parser.add_argument("--log-dir", default="results/live_session")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-bars", type=int, default=None, help="stop after N processed bars")
    parser.add_argument("--live", action="store_true", help="actually send orders (default: dry run)")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = replace(StrategyConfig(), risk_percent_initial_balance=args.risk)
    trader = LiveTrader(
        config=config,
        symbol=args.symbol,
        history_bars=args.history_bars,
        state_path=args.state,
        log_dir=args.log_dir,
        live=args.live,
        allow_real_money=args.i_understand_this_is_real_money,
        poll_seconds=args.poll_seconds,
    )

    try:
        trader.connect(args.terminal)
        trader.load_broker_spec()
        trader.bootstrap()
        trader.run(max_bars=args.max_bars)
    except LiveTraderError as exc:
        sys.exit(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
