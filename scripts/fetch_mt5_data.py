"""Export real XAUUSD M5 history and broker symbol specs from MetaTrader 5.

This is the Section 9.1 data path ("high-quality XAUUSD historical data with
realistic bid/ask or spread modelling") and the Section 5.3 requirement that
broker properties be read dynamically rather than assumed.

READ-ONLY. This script never places, modifies or closes an order. It calls
only MetaTrader5 data functions.

Credentials are read from the environment so they are never written to a
file or committed:

    $env:MT5_LOGIN    = "your_login"
    $env:MT5_SERVER   = "YourBroker-Demo"
    $env:MT5_PASSWORD = "..."          # never hard-code this
    python scripts/fetch_mt5_data.py --years 5 --out data/XAUUSD_M5_live.csv

If the terminal is already running and logged in, omit the credentials
entirely and the script attaches to that session instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - platform dependent
    sys.exit("MetaTrader5 package not installed. Run: pip install MetaTrader5")

GOLD_HINTS = ("XAUUSD", "GOLD", "XAU")


def connect(terminal_path: str | None) -> None:
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    kwargs = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    if login and password and server:
        kwargs.update(login=int(login), password=password, server=server)

    if not mt5.initialize(**kwargs):
        sys.exit(f"initialize() failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info is None:
        mt5.shutdown()
        sys.exit(f"not logged in: {mt5.last_error()}")
    modes = {0: "DEMO", 1: "CONTEST", 2: "REAL"}  # ENUM_ACCOUNT_TRADE_MODE
    mode = modes.get(info.trade_mode, f"UNKNOWN({info.trade_mode})")
    print(f"connected: login {info.login} on {info.server} ({info.company})")
    print(f"  balance {info.balance:,.2f} {info.currency}   account type: {mode}")
    if mode == "REAL":
        print("  *** THIS IS A LIVE-MONEY ACCOUNT ***")


def resolve_symbol(preferred: str | None) -> str:
    """Find the broker's gold symbol -- names vary (XAUUSD, XAUUSD.a, GOLD)."""
    if preferred:
        if mt5.symbol_select(preferred, True) and mt5.symbol_info(preferred):
            return preferred
        print(f"symbol {preferred!r} unavailable, searching...")

    candidates = [s.name for s in mt5.symbols_get() or [] if any(h in s.name.upper() for h in GOLD_HINTS)]
    if not candidates:
        mt5.shutdown()
        sys.exit("no gold symbol found on this account")
    candidates.sort(key=len)
    for name in candidates:
        if mt5.symbol_select(name, True):
            print(f"using symbol {name!r} (candidates: {', '.join(candidates[:8])})")
            return name
    mt5.shutdown()
    sys.exit(f"could not select any of: {candidates}")


def print_broker_spec(symbol: str) -> dict:
    """Section 5.3: these must drive BrokerSpec instead of assumed constants."""
    info = mt5.symbol_info(symbol)
    spec = {
        "symbol": symbol,
        "contract_size": info.trade_contract_size,
        "tick_size": info.trade_tick_size,
        "tick_value": info.trade_tick_value,
        "point": info.point,
        "digits": info.digits,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "spread_current_points": info.spread,
        "stops_level_points": info.trade_stops_level,
    }
    print("\n-- broker symbol specification (feed these into BrokerSpec) --")
    for key, value in spec.items():
        print(f"  {key:<24} {value}")
    return spec


def fetch(symbol: str, years: float, chunk_days: int = 30, timeframe: str = "M5") -> pd.DataFrame:
    """Pull M5 bars by walking backwards in chunks.

    A single copy_rates_range call stops returning data past roughly 180
    days, and copy_rates_from_pos is capped by the terminal's ``maxbars``.
    Walking backwards in windows sidesteps both limits; we stop when a
    window comes back empty or when the earliest bar stops receding, which
    is how the broker's actual history floor announces itself.
    """
    end = datetime.now(timezone.utc)
    floor = end - timedelta(days=int(365 * years))
    frames: list[pd.DataFrame] = []
    cursor = end
    previous_earliest = None

    while cursor > floor:
        window_start = max(cursor - timedelta(days=chunk_days), floor)
        tf = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
              "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30}[timeframe]
        rates = mt5.copy_rates_range(symbol, tf, window_start, cursor)
        if rates is None or len(rates) == 0:
            break

        chunk = pd.DataFrame(rates)
        frames.append(chunk)
        earliest = chunk["time"].min()
        if previous_earliest is not None and earliest >= previous_earliest:
            break  # broker clamped us to its history floor
        previous_earliest = earliest
        cursor = window_start

    if not frames:
        mt5.shutdown()
        sys.exit(f"no M5 bars returned: {mt5.last_error()}  (open a {symbol} M5 chart to force history download)")

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], unit="s")
    frame = frame.rename(columns={"tick_volume": "volume"})
    return frame[["time", "open", "high", "low", "close", "volume", "spread"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--years", type=float, default=5.0, help="the D1 EMA200 alone needs 200 trading days")
    parser.add_argument("--out", default="data/XAUUSD_M5_live.csv")
    parser.add_argument("--timeframe", default="M5", choices=("M1", "M5", "M15", "M30"))
    parser.add_argument("--terminal", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    args = parser.parse_args()

    connect(args.terminal)
    try:
        symbol = resolve_symbol(args.symbol)
        spec = print_broker_spec(symbol)
        frame = fetch(symbol, args.years, timeframe=args.timeframe)
    finally:
        mt5.shutdown()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame.to_csv(args.out, index=False)

    span_days = (frame["time"].max() - frame["time"].min()).days
    print(f"\nwrote {len(frame):,} M5 bars to {args.out}")
    print(f"  range   {frame['time'].min()}  ->  {frame['time'].max()}  ({span_days} days)")
    print(f"  spread  median {frame['spread'].median():.0f} points, p95 {frame['spread'].quantile(0.95):.0f} points")
    if len(frame) < 57600:
        print("  WARNING: under ~57,600 bars the D1 EMA200 never warms up -- fetch more history.")


if __name__ == "__main__":
    main()
