"""Run the strategy across several symbols on one account.

Each symbol gets its own engines -- zone detection, state machine, safety
counters and broker specification -- because the rules are per-instrument and
a setup on one symbol says nothing about another. What they share is the
account, so a single PortfolioRisk sits above them all and decides whether an
otherwise valid entry may actually open (see xauusd_robot/portfolio.py).

The MetaTrader5 package holds one process-wide connection, so the terminal is
initialised once here and every symbol borrows it.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from typing import Dict, List, Optional

from .adaptive import DEFAULT_EXPECTATION, PerformanceTracker, TradeLedger
from .config import SYMBOL_SPECS, StrategyConfig, live_config
from .live import TRADE_MODES, LiveTrader, LiveTraderError
from .portfolio import PortfolioLimits, PortfolioRisk, PortfolioState


class MultiSymbolTrader:
    def __init__(
        self,
        symbols: List[str],
        risk_percent: float = 2.0,
        history_bars: int = 60000,
        state_dir: str = "state",
        log_dir: str = "results/live_session",
        live: bool = False,
        allow_real_money: bool = False,
        poll_seconds: int = 10,
        limits: Optional[PortfolioLimits] = None,
        execution_tf: str = "M5",
    ):
        self.symbols = list(symbols)
        self.risk_percent = risk_percent
        self.history_bars = history_bars
        self.state_dir = state_dir
        self.log_dir = log_dir
        self.live = live
        self.allow_real_money = allow_real_money
        self.poll_seconds = poll_seconds
        self.limits = limits or PortfolioLimits()
        self.execution_tf = execution_tf

        self.traders: Dict[str, LiveTrader] = {}
        self.portfolio: Optional[PortfolioRisk] = None
        self.ledger = TradeLedger(os.path.join(state_dir, "trade_ledger.json")).load()
        self.tracker = PerformanceTracker(self.ledger, DEFAULT_EXPECTATION)
        self.status = "idle"
        self.log_lines: List[str] = []
        self.mt5 = None

    # ------------------------------------------------------------------
    def _portfolio_path(self) -> str:
        return os.path.join(self.state_dir, "portfolio.json")

    def _load_portfolio(self, balance: float) -> PortfolioRisk:
        path = self._portfolio_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    state = PortfolioState.from_json(fh.read())
                print(f"  portfolio    : resumed (peak {state.running_equity_peak:,.2f}, "
                      f"dd_lock={state.drawdown_locked}, target={state.target_reached})")
                return PortfolioRisk(state, self.limits)
            except (OSError, ValueError, TypeError):
                pass
        print(f"  portfolio    : new, initial balance reference {balance:,.2f}")
        return PortfolioRisk(PortfolioState.new(balance), self.limits)

    def persist_portfolio(self) -> None:
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with open(self._portfolio_path(), "w", encoding="utf-8") as fh:
                fh.write(self.portfolio.state.to_json())
        except OSError:
            pass

    # ------------------------------------------------------------------
    def connect(self, terminal_path: Optional[str] = None) -> None:
        """Initialise the terminal once, then build one trader per symbol."""
        primary = self._make_trader(self.symbols[0])
        primary.connect(terminal_path)  # validates demo/autotrading, sets account
        self.mt5 = primary.mt5
        account = primary.account
        self.portfolio = self._load_portfolio(account.balance)

        primary.portfolio = self.portfolio
        primary.ledger = self.ledger
        self.traders[self.symbols[0]] = primary

        for symbol in self.symbols[1:]:
            trader = self._make_trader(symbol)
            # The MetaTrader5 connection is process-wide, so the remaining
            # traders reuse it rather than re-initialising the terminal.
            trader.account = account
            trader.portfolio = self.portfolio
            trader.ledger = self.ledger
            self.traders[symbol] = trader

    def _make_trader(self, symbol: str) -> LiveTrader:
        config = replace(
            live_config(self.risk_percent, SYMBOL_SPECS.get(symbol)),
            symbol=symbol,
            execution_tf=self.execution_tf,
        )
        return LiveTrader(
            config=config,
            symbol=symbol,
            history_bars=self.history_bars,
            state_path=os.path.join(self.state_dir, f"live_state_{symbol}.json"),
            log_dir=self.log_dir,
            live=self.live,
            allow_real_money=self.allow_real_money,
            poll_seconds=self.poll_seconds,
        )

    def bootstrap(self) -> None:
        ready, failed = [], []
        for symbol, trader in self.traders.items():
            try:
                print(f"\n--- {symbol} ---")
                trader.load_broker_spec()
                trader.bootstrap()
                ready.append(symbol)
            except Exception as exc:
                print(f"  {symbol}: UNAVAILABLE ({type(exc).__name__}: {exc})")
                failed.append(symbol)
        for symbol in failed:
            self.traders.pop(symbol, None)
        if not self.traders:
            raise LiveTraderError("no symbols could be initialised")
        print(f"\ntrading {len(ready)} symbol(s): {', '.join(ready)}")
        if failed:
            print(f"skipped: {', '.join(failed)}")

    # ------------------------------------------------------------------
    def set_live(self, live: bool) -> str:
        for trader in self.traders.values():
            error = trader.set_live(live)
            if error:
                # Roll back so symbols cannot end up in mixed modes.
                for other in self.traders.values():
                    other.live = False
                return error
        self.live = live
        return ""

    def snapshot(self) -> dict:
        symbols = {}
        for symbol, trader in self.traders.items():
            try:
                symbols[symbol] = trader.snapshot()
            except Exception as exc:
                symbols[symbol] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        account = None
        try:
            info = self.mt5.account_info()
            if info:
                peak = self.portfolio.state.running_equity_peak or 1.0
                account = {
                    "login": info.login, "server": info.server,
                    "type": TRADE_MODES.get(info.trade_mode, "UNKNOWN"),
                    "balance": round(info.balance, 2), "equity": round(info.equity, 2),
                    "currency": info.currency,
                    "drawdown_percent": round(max(0.0, (1 - info.equity / peak) * 100), 2),
                }
        except Exception:
            pass

        return {
            "mode": "LIVE" if self.live else "DRY_RUN",
            "status": self.status,
            "symbols": symbols,
            "account": account,
            "portfolio": self.portfolio.snapshot() if self.portfolio else {},
            "learning": self.tracker.assess(),
            "precision_table": self.tracker.precision_table(),
            "risk_percent": self.risk_percent,
            "execution_tf": self.execution_tf,
        }

    # ------------------------------------------------------------------
    def run(self, stop_event: Optional[threading.Event] = None, max_cycles: Optional[int] = None) -> None:
        self.status = "running"
        print(f"\n--- multi-symbol loop started ({len(self.traders)} symbols) ---")
        cycles = 0
        try:
            while max_cycles is None or cycles < max_cycles:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    account = self.mt5.account_info()
                    if account:
                        self.portfolio.update_equity(account.equity)

                    for symbol, trader in self.traders.items():
                        try:
                            latest = trader._fetch_bars(3)
                            newest_closed = latest.index[-2]
                            if trader.last_bar_time is None or newest_closed > trader.last_bar_time:
                                self.portfolio.roll_day(str(newest_closed.date()))
                                with trader._lock:
                                    trader.on_new_bar()
                                trader._log(
                                    f"bar {trader.last_bar_time} | "
                                    f"regime {trader._regime_at(len(trader.bars) - 1).value} | "
                                    f"state {trader.sm.state.value}"
                                )
                        except Exception as exc:
                            trader._log(f"WARNING: {type(exc).__name__}: {exc}")
                    self.persist_portfolio()
                    cycles += 1
                except Exception as exc:
                    print(f"loop warning: {type(exc).__name__}: {exc}", flush=True)

                for _ in range(max(1, self.poll_seconds)):
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n--- stopped by user ---")
        finally:
            self.status = "stopped"
            self.persist_portfolio()
            for trader in self.traders.values():
                trader.persist_state()
