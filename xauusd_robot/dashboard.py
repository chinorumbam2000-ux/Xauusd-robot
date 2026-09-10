"""Localhost monitoring and control dashboard (Blueprint Section 12).

Runs the LiveTrader in a background thread and serves a single-page UI plus
a small JSON API. Uses only the standard library, so there is no new
dependency and nothing is exposed beyond the loopback interface.

Panels follow the blueprint's "Recommended Demo Dashboard": six EMA200
regime lights, active zones and confluence, WPR(49) state, setup state
machine, risk budget and sizing, safety/circuit-breaker status, open
position, and a live event log.

Binds to 127.0.0.1 only. Control endpoints mutate a live trading session,
so this must never be exposed to a network.
"""
from __future__ import annotations

import json
import os
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from .live import LiveTrader, LiveTraderError

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(HERE, "dashboard.html")


class Supervisor:
    """Owns the LiveTrader and its background thread."""

    def __init__(self, trader: LiveTrader):
        self.trader = trader
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> str:
        if self.running:
            return "already running"
        self.stop_event = threading.Event()
        kwargs = {"stop_event": self.stop_event}
        if isinstance(self.trader, LiveTrader):
            # The dashboard owns the terminal connection; the loop must not
            # close it when a single-symbol trader stops.
            kwargs["shutdown_on_exit"] = False
        self.thread = threading.Thread(
            target=self.trader.run, kwargs=kwargs, daemon=True, name="live-trader",
        )
        self.thread.start()
        return ""

    def stop(self) -> str:
        if not self.running:
            return "not running"
        self.stop_event.set()
        self.thread.join(timeout=15)
        return ""

    def snapshot(self) -> dict:
        snap = self.trader.snapshot()
        snap["running"] = self.running
        return snap


class Handler(BaseHTTPRequestHandler):
    supervisor: Supervisor = None  # injected

    def log_message(self, fmt, *args):  # silence per-request stderr spam
        pass

    # -- helpers -------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- multi-symbol helpers ------------------------------------------
    def _symbol_trader(self, symbol):
        trader = self.supervisor.trader
        if isinstance(trader, LiveTrader):
            return trader if (symbol in (None, trader.symbol)) else None
        return trader.traders.get(symbol) if symbol else None

    def _reset(self, which: str) -> str:
        """Clear a lock at whichever level owns it."""
        trader = self.supervisor.trader
        targets = [trader] if isinstance(trader, LiveTrader) else list(trader.traders.values())
        for t in targets:
            if which == "drawdown":
                t.safety.manual_reset_drawdown_lock()
            else:
                t.safety.manual_reset_target()
            t.persist_state()
            t._log(f"{which} lock manually reset from dashboard")
        portfolio = getattr(trader, "portfolio", None)
        if portfolio is not None:
            if which == "drawdown":
                portfolio.reset_drawdown_lock()
            else:
                portfolio.reset_target()
            trader.persist_portfolio()
        return ""

    # -- routes --------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                with open(PAGE_PATH, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"dashboard.html missing", "text/plain")
        elif path == "/api/status":
            self._json(self.supervisor.snapshot())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        trader = self.supervisor.trader
        body = self._read_json()

        if path == "/api/start":
            error = self.supervisor.start()
        elif path == "/api/stop":
            error = self.supervisor.stop()
        elif path == "/api/mode":
            error = trader.set_live(bool(body.get("live")))
        elif path == "/api/risk":
            try:
                from dataclasses import replace

                value = float(body.get("risk_percent"))
                if not 0 < value <= 10:
                    raise ValueError
                targets = ([trader] if isinstance(trader, LiveTrader)
                           else list(trader.traders.values()))
                for t in targets:
                    t.config = replace(t.config, risk_percent_initial_balance=value)
                    t.safety.config = t.config
                    t._log(f"risk set to {value}% of initial balance")
                if not isinstance(trader, LiveTrader):
                    trader.risk_percent = value
                error = ""
            except (TypeError, ValueError):
                error = "risk_percent must be a number between 0 and 10"
        elif path == "/api/close-position":
            symbol = body.get("symbol")
            target = self._symbol_trader(symbol)
            error = target.close_position() if target else f"unknown symbol {symbol!r}"
        elif path == "/api/reset-drawdown":
            error = self._reset("drawdown")
        elif path == "/api/reset-target":
            error = self._reset("target")
        else:
            self._send(404, b"not found", "text/plain")
            return

        self._json({"ok": not error, "error": error})


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def serve(trader: LiveTrader, host: str = "127.0.0.1", port: int = 8765, autostart: bool = True) -> None:
    # HTTPServer sets allow_reuse_address, which on Windows lets a second
    # instance bind a port another process is already serving. The result is
    # two robots against one account and a UI showing whichever answers
    # first, so refuse up front instead.
    if _port_in_use(host, port):
        raise LiveTraderError(
            f"port {port} is already serving -- another dashboard is running.\n"
            f"  Stop it first, or pass --port with a free port."
        )

    supervisor = Supervisor(trader)
    if autostart:
        supervisor.start()

    Handler.supervisor = supervisor
    server = ThreadingHTTPServer((host, port), Handler)

    print(f"\n  dashboard : http://{host}:{port}")
    print("  bound to loopback only -- do not expose this to a network")
    print("  Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        supervisor.stop()
        server.server_close()
        trader.mt5.shutdown()
