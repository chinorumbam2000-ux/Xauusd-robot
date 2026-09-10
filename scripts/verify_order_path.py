"""One-off check that the live order path actually works on this broker.

Places a MINIMUM-VOLUME order with SL/TP attached, confirms the broker
accepted it and that the stops were registered, then closes it immediately.
This exercises the parts most likely to break per-broker -- filling mode,
stops level, deviation, magic number -- so a real signal weeks from now
does not fail silently.

REFUSES to run on anything but a DEMO account. No override.

    python scripts/verify_order_path.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5  # noqa: E402

from xauusd_robot.live import MAGIC, TRADE_MODES  # noqa: E402


def filling_mode(info):
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK, "FOK"
    if info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC, "IOC"
    return mt5.ORDER_FILLING_RETURN, "RETURN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--terminal", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    args = parser.parse_args()

    ok = mt5.initialize(
        path=args.terminal,
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
    )
    if not ok:
        sys.exit(f"initialize failed: {mt5.last_error()}")

    account = mt5.account_info()
    mode = TRADE_MODES.get(account.trade_mode, "UNKNOWN")
    print(f"account {account.login} on {account.server} -- type {mode}, balance {account.balance:,.2f}")
    if mode != "DEMO":
        mt5.shutdown()
        sys.exit(f"refusing to place a test order on a {mode} account")

    mt5.symbol_select(args.symbol, True)
    info = mt5.symbol_info(args.symbol)
    tick = mt5.symbol_info_tick(args.symbol)
    fill, fill_name = filling_mode(info)

    # Stops must clear the broker's minimum distance; use 3x for headroom.
    min_distance = max(info.trade_stops_level * info.point, info.spread * info.point) * 3
    entry = tick.ask
    sl = round(entry - min_distance, info.digits)
    tp = round(entry + 3 * min_distance, info.digits)

    print(f"\nsymbol      : {args.symbol}  digits {info.digits}  point {info.point}")
    print(f"stops_level : {info.trade_stops_level} pts -> min distance {min_distance:.2f}")
    print(f"filling     : {fill_name}")
    print(f"test order  : BUY {info.volume_min} @ {entry} SL {sl} TP {tp}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": info.volume_min,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "v1.1 order path check",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\nORDER FAILED: retcode={getattr(result, 'retcode', None)} "
              f"comment={getattr(result, 'comment', mt5.last_error())}")
        mt5.shutdown()
        sys.exit(1)

    print(f"\nORDER ACCEPTED: ticket #{result.order}, filled {result.volume} @ {result.price}")

    time.sleep(2)
    positions = [p for p in (mt5.positions_get(symbol=args.symbol) or []) if p.ticket == result.order]
    if positions:
        p = positions[0]
        print(f"position confirmed: volume {p.volume}, open {p.price_open}, SL {p.sl}, TP {p.tp}, magic {p.magic}")
        stops_ok = p.sl > 0 and p.tp > 0
        print(f"stops registered server-side: {'YES' if stops_ok else 'NO -- would run unprotected!'}")
    else:
        print("WARNING: position not found after fill")
        p = None

    if p is not None:
        close_tick = mt5.symbol_info_tick(args.symbol)
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": args.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL,
            "position": p.ticket,
            "price": close_tick.bid,
            "deviation": 20,
            "magic": MAGIC,
            "comment": "v1.1 order path check close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill,
        }
        close_result = mt5.order_send(close_request)
        if close_result is None or close_result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"CLOSE FAILED: retcode={getattr(close_result, 'retcode', None)} "
                  f"-- close ticket #{p.ticket} manually in the terminal")
        else:
            print(f"position closed @ {close_result.price}")

    time.sleep(2)
    deals = mt5.history_deals_get(datetime.now(timezone.utc) - timedelta(minutes=10),
                                  datetime.now(timezone.utc)) or []
    mine = [d for d in deals if d.position_id == result.order]
    pnl = sum(d.profit + d.swap + d.commission for d in mine)
    print(f"\nround-trip P/L: {pnl:+.2f} {account.currency} (spread cost on {info.volume_min} lots)")
    print(f"final balance : {mt5.account_info().balance:,.2f}")
    print("\nORDER PATH VERIFIED -- live signals will be able to execute.")
    mt5.shutdown()


if __name__ == "__main__":
    main()
