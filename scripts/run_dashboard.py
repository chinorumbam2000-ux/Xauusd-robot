"""Launch the localhost monitoring/control dashboard (Blueprint Section 12).

    $env:MT5_LOGIN="..."; $env:MT5_SERVER="Broker-Demo"; $env:MT5_PASSWORD="..."
    python scripts/run_dashboard.py                       # XAUUSD only
    python scripts/run_dashboard.py --symbols XAUUSD EURUSD GBPUSD USDJPY EURGBP

Then open http://127.0.0.1:8765

The robot starts in DRY RUN. Arming live order placement is a button in the
UI, which refuses non-demo accounts and refuses while the terminal's
AutoTrading switch is off.

Default symbol set is XAUUSD alone, because it is the only instrument measured
with a positive edge (see scripts/symbol_comparison.py). The others can be
enabled explicitly; the README records what each one actually returned.

The server binds to loopback only. Its control endpoints mutate a live trading
session, so never expose it to a network.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xauusd_robot.config import TRADEABLE_SYMBOLS  # noqa: E402
from xauusd_robot.dashboard import serve  # noqa: E402
from xauusd_robot.live import LiveTraderError  # noqa: E402
from xauusd_robot.multi_symbol import MultiSymbolTrader  # noqa: E402
from xauusd_robot.portfolio import PortfolioLimits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["XAUUSD"], choices=list(TRADEABLE_SYMBOLS))
    parser.add_argument("--risk", type=float, default=2.0, help="percent of INITIAL balance per trade")
    parser.add_argument("--execution-tf", default="M5", choices=("M1", "M5", "M15"),
                        help="entry timeframe; M1 trades far more often but tested at ~zero edge")
    parser.add_argument("--history-bars", type=int, default=60000)
    parser.add_argument("--terminal", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--log-dir", default="results/live_session")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-positions", type=int, default=2, help="account-wide open position cap")
    parser.add_argument("--max-per-cluster", type=int, default=1,
                        help="open positions allowed within one correlated group")
    parser.add_argument("--observe", action="store_true",
                        help="DEMO ONLY: disable the Section 5.5 circuit breakers so the machinery "
                             "can be watched without halting itself. Stops stay on every order.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--live", action="store_true", help="start already armed (default: dry run)")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        print("WARNING: binding outside loopback exposes live trading controls to the network.")

    limits = PortfolioLimits(
        max_total_positions=args.max_positions,
        max_positions_per_cluster=args.max_per_cluster,
    )
    trader = MultiSymbolTrader(
        symbols=args.symbols,
        risk_percent=args.risk,
        history_bars=args.history_bars,
        state_dir=args.state_dir,
        log_dir=args.log_dir,
        live=args.live,
        allow_real_money=args.i_understand_this_is_real_money,
        poll_seconds=args.poll_seconds,
        limits=limits,
        execution_tf=args.execution_tf,
        observation_mode=args.observe,
    )

    try:
        trader.connect(args.terminal)
        trader.bootstrap()
    except LiveTraderError as exc:
        sys.exit(f"\nERROR: {exc}")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)
    try:
        serve(trader, host=args.host, port=args.port)
    except LiveTraderError as exc:
        sys.exit(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
