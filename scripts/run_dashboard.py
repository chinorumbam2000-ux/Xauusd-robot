"""Launch the localhost monitoring/control dashboard (Blueprint Section 12).

    $env:MT5_LOGIN="..."; $env:MT5_SERVER="Broker-Demo"; $env:MT5_PASSWORD="..."
    python scripts/run_dashboard.py

Then open http://127.0.0.1:8765

The robot starts in DRY RUN. Arm live order placement from the UI button,
which refuses non-demo accounts and refuses to arm while the terminal's
AutoTrading switch is off.

The server binds to loopback only. Its control endpoints mutate a live
trading session, so never expose it to a network.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import StrategyConfig  # noqa: E402
from xauusd_robot.dashboard import serve  # noqa: E402
from xauusd_robot.live import LiveTrader, LiveTraderError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--history-bars", type=int, default=60000)
    parser.add_argument("--terminal", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--state", default="state/live_state.json")
    parser.add_argument("--log-dir", default="results/live_session")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--live", action="store_true", help="start already armed (default: dry run)")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        print("WARNING: binding outside loopback exposes live trading controls to the network.")

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
    except LiveTraderError as exc:
        sys.exit(f"\nERROR: {exc}")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)
    serve(trader, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
