"""Live / demo forward-testing bridge (Blueprint Phase 6).

Runs the *same* rule engines as the backtester against a live MetaTrader 5
terminal: identical regime filter, zone engine, state machine, risk engine
and safety layer. The only differences are that bars arrive one at a time
and that orders are real.

Safety design
-------------
* Refuses to run on a REAL-money account unless explicitly overridden. The
  MT5 trade-mode enum is 0=DEMO, 1=CONTEST, 2=REAL -- easy to get backwards,
  so the check is written against names, never raw integers.
* Defaults to dry-run. Orders are only sent when ``live=True``.
* Broker symbol properties (tick size/value, volume min/step/max, stops
  level) are read from the terminal, per Section 5.3 -- never assumed.
* Stop-loss and take-profit are attached to the order itself, so the broker
  enforces them even if this process dies.
* Safety state is persisted every bar, so a restart preserves the initial
  balance reference, running equity peak and every lock (Section 12).
* Only positions carrying this robot's magic number are ever managed.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from .config import BrokerSpec, StrategyConfig
from .data import build_regime_frame, build_timeframe_indicators
from .indicators import add_core_indicators
from .logging_engine import RunLogger
from .regime import Regime, compute_regime
from .risk import compute_lot_size, compute_stop_loss
from .safety import SafetyEngine, SafetyState
from .state_machine import SetupStateMachine
from .zones import ZoneEngine

TRADE_MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}
MAGIC = 20260910


class LiveTraderError(RuntimeError):
    pass


class LiveTrader:
    def __init__(
        self,
        config: StrategyConfig,
        symbol: str = "XAUUSD",
        history_bars: int = 60000,
        state_path: str = "state/live_state.json",
        log_dir: str = "results/live_session",
        live: bool = False,
        allow_real_money: bool = False,
        poll_seconds: int = 10,
    ):
        self.config = config
        self.symbol = symbol
        self.history_bars = history_bars
        self.state_path = state_path
        self.log_dir = log_dir
        self.live = live
        self.allow_real_money = allow_real_money
        self.poll_seconds = poll_seconds

        self.mt5 = self._import_mt5()
        self.logger = RunLogger()
        self.bars: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None
        self.open_ticket: Optional[int] = None
        self.open_risk_money: float = 0.0

        # Dashboard support: a rolling log tail and a coarse status flag so a
        # supervising UI can render what the loop is doing without scraping stdout.
        self.log_lines: deque = deque(maxlen=400)
        self.status: str = "idle"
        self.account = None
        self.symbol_info = None
        self._lock = threading.Lock()

    def _log(self, message: str) -> None:
        line = f"[{self._now()}] {message}"
        print(line, flush=True)
        self.log_lines.append(line)

    @staticmethod
    def _import_mt5():
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise LiveTraderError("MetaTrader5 package not installed (pip install MetaTrader5)") from exc
        return mt5

    # ------------------------------------------------------------------
    def connect(self, terminal_path: Optional[str] = None) -> None:
        kwargs = {}
        if terminal_path:
            kwargs["path"] = terminal_path
        login, password, server = (
            os.environ.get("MT5_LOGIN"), os.environ.get("MT5_PASSWORD"), os.environ.get("MT5_SERVER")
        )
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)

        if not self.mt5.initialize(**kwargs):
            raise LiveTraderError(f"initialize() failed: {self.mt5.last_error()}")

        account = self.mt5.account_info()
        if account is None:
            raise LiveTraderError(f"not logged in: {self.mt5.last_error()}")

        mode = TRADE_MODES.get(account.trade_mode, f"UNKNOWN({account.trade_mode})")
        self.account = account
        print(f"connected: login {account.login} on {account.server} ({account.company})")
        print(f"  account type : {mode}")
        print(f"  balance      : {account.balance:,.2f} {account.currency}")
        print(f"  mode         : {'LIVE ORDER PLACEMENT' if self.live else 'DRY RUN (no orders sent)'}")

        if mode != "DEMO" and not self.allow_real_money:
            self.mt5.shutdown()
            raise LiveTraderError(
                f"refusing to trade a {mode} account. This robot has not completed the "
                "out-of-sample, walk-forward and forward-testing gates the blueprint requires "
                "before real capital (Section 11.2). Override only with deliberate intent."
            )
        if not account.trade_allowed:
            raise LiveTraderError("trading is disabled for this account by the broker")

        # The AutoTrading toolbar switch is a deliberate client-side control with
        # no API to enable it. Fail fast here rather than at signal time, weeks
        # later, with retcode 10027.
        terminal = self.mt5.terminal_info()
        if self.live and not terminal.trade_allowed:
            self.mt5.shutdown()
            raise LiveTraderError(
                "AutoTrading is switched OFF in the MetaTrader 5 terminal, so every order "
                "would be rejected with retcode 10027.\n"
                "  Fix: in the MT5 terminal, click the 'Algo Trading' (AutoTrading) toolbar "
                "button so it turns green, or Tools > Options > Expert Advisors > "
                "'Allow algorithmic trading'.\n"
                "  Then re-run this command."
            )
        if self.live:
            print("  autotrading  : ENABLED")

    def load_broker_spec(self) -> BrokerSpec:
        """Section 5.3: read the specification, never assume it."""
        if not self.mt5.symbol_select(self.symbol, True):
            raise LiveTraderError(f"cannot select symbol {self.symbol}")
        info = self.mt5.symbol_info(self.symbol)
        spec = BrokerSpec(
            symbol=self.symbol,
            contract_size=info.trade_contract_size,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            volume_step=info.volume_step,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            leverage=float(self.account.leverage or 100),
            default_spread_points=float(info.spread or 15),
        )
        self.symbol_info = info
        self.stops_level_points = info.trade_stops_level
        self.config = replace(self.config, broker=spec)
        print(f"  broker spec  : tick {spec.tick_size} / value {spec.tick_value} / "
              f"vol {spec.volume_min}-{spec.volume_max} step {spec.volume_step} / "
              f"stops_level {self.stops_level_points} pts")
        return spec

    # ------------------------------------------------------------------
    def _fetch_bars(self, count: int) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M5, 0, count)
        if rates is None or len(rates) == 0:
            raise LiveTraderError(f"no M5 bars: {self.mt5.last_error()}")
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        frame = frame.set_index("time").rename(columns={"tick_volume": "volume"})
        return frame[["open", "high", "low", "close", "volume", "spread"]]

    def _drop_forming_bar(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The newest bar from MT5 is still forming -- decisions use closed bars only."""
        return frame.iloc[:-1]

    def _enrich(self, raw: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        bars = add_core_indicators(raw, cfg.ema_period, cfg.atr_period, cfg.wpr_period)
        tf_frames = build_timeframe_indicators(raw, cfg)
        self.regime_frame = build_regime_frame(bars, tf_frames, cfg)
        self.regime_series = compute_regime(self.regime_frame, cfg.regime_timeframes)
        return bars

    def bootstrap(self) -> None:
        """Load history and replay it so zone/setup state is continuous."""
        raw = self._drop_forming_bar(self._fetch_bars(self.history_bars))
        self.raw = raw
        self.bars = self._enrich(raw)
        self.last_bar_time = self.bars.index[-1]

        state = self._load_state()
        self.safety = SafetyEngine(self.config, state)
        self.zone_engine = ZoneEngine(self.config, self.bars)
        self.sm = SetupStateMachine(self.config, self.bars)

        print(f"  history      : {len(self.bars):,} closed M5 bars "
              f"({self.bars.index[0]} -> {self.bars.index[-1]})")
        print("  replaying history to rebuild zone state...")
        for i in range(len(self.bars)):
            events = self.zone_engine.process_bar(i)
            regime = self._regime_at(i)
            if self.sm.setup is not None:
                self.sm.update(i, regime)
            if self.sm.is_idle and regime is not Regime.MIXED:
                for event in events:
                    if event.direction == regime.value:
                        self.sm.arm(event)
                        break
        print(f"  state        : {self.sm.state.value}, "
              f"{len(self.zone_engine.active_zones):,} live zones")
        self.reconcile_positions()

    def _regime_at(self, i: int) -> Regime:
        value = self.regime_series.iloc[i]
        return value if isinstance(value, Regime) else Regime(value)

    # ------------------------------------------------------------------
    def _load_state(self) -> SafetyState:
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as fh:
                state = SafetyState.from_json(fh.read())
            print(f"  state file   : resumed from {self.state_path} "
                  f"(peak {state.running_equity_peak:,.2f}, "
                  f"locks dd={state.drawdown_locked} target={state.target_reached})")
            return state
        state = SafetyState.new(self.account.balance)
        print(f"  state file   : new, initial balance reference {state.initial_balance:,.2f}")
        return state

    def persist_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as fh:
            fh.write(self.safety.state.to_json())

    def reconcile_positions(self) -> None:
        """Adopt any position this robot already owns (restart safety)."""
        positions = self.mt5.positions_get(symbol=self.symbol) or []
        mine = [p for p in positions if p.magic == MAGIC]
        if mine:
            position = mine[0]
            self.open_ticket = position.ticket
            self.safety.state.open_positions = 1
            self.sm.on_trade_opened()
            print(f"  reconciled   : adopted open position #{position.ticket} "
                  f"({position.volume} lots @ {position.price_open})")
        else:
            self.open_ticket = None
            self.safety.state.open_positions = 0

    def sync_closed_position(self) -> None:
        """Detect a broker-side SL/TP close and fold it into the safety state."""
        if self.open_ticket is None:
            return
        still_open = [
            p for p in (self.mt5.positions_get(symbol=self.symbol) or [])
            if p.ticket == self.open_ticket
        ]
        if still_open:
            return

        deals = self.mt5.history_deals_get(
            datetime.now(timezone.utc) - pd.Timedelta(days=7), datetime.now(timezone.utc)
        ) or []
        closing = [d for d in deals if d.position_id == self.open_ticket and d.entry == self.mt5.DEAL_ENTRY_OUT]
        profit = sum(d.profit + d.swap + d.commission for d in closing)

        risk_money = getattr(self, "open_risk_money", 0.0) or 1.0
        r_multiple = profit / risk_money
        balance = self.mt5.account_info().balance
        bar_index = len(self.bars) - 1

        self.safety.register_trade_close(r_multiple, balance, bar_index)
        self.sm.on_trade_closed()
        self._log(f"position #{self.open_ticket} closed: {profit:+,.2f} "
                  f"({r_multiple:+.2f}R), balance {balance:,.2f}")
        self.logger.log_trade(
            exit_time=pd.Timestamp.utcnow(), ticket=self.open_ticket,
            pnl_money=profit, r_multiple=r_multiple, balance_after=balance,
        )
        self.open_ticket = None
        self.persist_state()

    # ------------------------------------------------------------------
    def on_new_bar(self) -> None:
        raw = self._drop_forming_bar(self._fetch_bars(self.history_bars))
        if raw.index[-1] <= self.last_bar_time:
            return

        self.raw = raw
        self.bars = self._enrich(raw)
        self.zone_engine.update_bars(self.bars)
        self.sm.update_bars(self.bars)

        i = len(self.bars) - 1
        self.last_bar_time = self.bars.index[i]

        self.safety.roll_broker_day(self.last_bar_time.date())
        self.sync_closed_position()

        account = self.mt5.account_info()
        self.safety.update_equity(account.equity)

        events = self.zone_engine.process_bar(i)
        regime = self._regime_at(i)

        if self.safety.state.target_reached or self.safety.state.drawdown_locked:
            if self.sm.setup is not None:
                self.sm.cancel()
            self.persist_state()
            return

        if self.sm.setup is not None:
            outcome = self.sm.update(i, regime)
            if outcome.kind == "entry_ready":
                self.try_entry(i, outcome.setup)
            elif outcome.kind in ("expired", "zone_invalidated"):
                self.logger.log_event(time=self.last_bar_time, event="setup_cancelled", reason=outcome.kind)

        if self.sm.is_idle and self.open_ticket is None and regime is not Regime.MIXED:
            if i >= self.safety.state.cooldown_until_bar:
                for event in events:
                    if event.direction == regime.value:
                        self.sm.arm(event)
                        self._log(f"setup armed: {event.direction} from {event.zone.type.value} "
                                  f"zone [{event.zone.low:.2f}, {event.zone.high:.2f}], "
                                  f"confluence {event.confluence_score}")
                        break

        self.persist_state()

    # ------------------------------------------------------------------
    def try_entry(self, i: int, setup) -> None:
        cfg = self.config
        tick = self.mt5.symbol_info_tick(self.symbol)
        spread = tick.ask - tick.bid
        atr_value = float(self.bars["atr"].iloc[i])
        entry = tick.ask if setup.direction == "BUY" else tick.bid

        sl = compute_stop_loss(
            setup.direction, setup.zone.far_boundary, setup.reaction_extreme,
            atr_value, spread, cfg.sl_buffer_atr, cfg.sl_buffer_spread_mult,
        )
        if setup.direction == "BUY":
            stop_price = sl.stop_price
            stop_distance = entry - stop_price
            target_price = entry + cfg.reward_risk * stop_distance
        else:
            stop_price = sl.stop_price + spread
            stop_distance = stop_price - entry
            target_price = entry - cfg.reward_risk * stop_distance

        def reject(reason: str) -> None:
            self._log(f"entry REJECTED ({reason})")
            self.logger.log_event(time=self.last_bar_time, event="entry_rejected", reason=reason,
                                  direction=setup.direction, entry=entry, stop=stop_price)
            self.sm.cancel()

        if stop_distance <= 0:
            return reject("invalid_stop_distance")

        allowed, reason = self.safety.can_open_new_trade(i, spread, stop_distance, atr_value)
        if not allowed:
            return reject(reason)

        # Broker minimum stop distance (SYMBOL_TRADE_STOPS_LEVEL).
        min_stop = self.stops_level_points * self.symbol_info.point
        if stop_distance < min_stop:
            return reject(f"stop_closer_than_broker_minimum({min_stop:.2f})")

        risk_budget = self.safety.state.initial_balance * cfg.risk_fraction()
        lot = compute_lot_size(stop_distance, risk_budget, cfg.broker)
        if not lot.accepted:
            return reject(lot.reason)

        self.place_order(setup, lot, entry, stop_price, target_price, stop_distance, spread)

    def place_order(self, setup, lot, entry, stop_price, target_price, stop_distance, spread) -> None:
        digits = self.symbol_info.digits
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(lot.normalized_lots, 2),
            "type": self.mt5.ORDER_TYPE_BUY if setup.direction == "BUY" else self.mt5.ORDER_TYPE_SELL,
            "price": round(entry, digits),
            "sl": round(stop_price, digits),
            "tp": round(target_price, digits),
            "deviation": 20,
            "magic": MAGIC,
            "comment": f"v1.1 {setup.zone.type.value} c{setup.confluence_score}",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        summary = (f"{setup.direction} {request['volume']} lots @ {request['price']} "
                   f"SL {request['sl']} TP {request['tp']} "
                   f"(risk {lot.normalized_risk:,.2f}, {stop_distance:.2f} stop)")

        if not self.live:
            self._log(f"DRY RUN would place: {summary}")
            self.logger.log_event(time=self.last_bar_time, event="dry_run_order", reason="ok", **request)
            self.sm.cancel()
            return

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            code = "none" if result is None else result.retcode
            comment = "" if result is None else result.comment
            self._log(f"ORDER FAILED (retcode {code} {comment}): {summary}")
            self.logger.log_event(time=self.last_bar_time, event="order_failed",
                                  reason=str(code), comment=comment, **request)
            self.sm.cancel()
            return

        self.open_ticket = result.order
        self.open_risk_money = lot.normalized_risk
        self.safety.register_trade_open()
        self.sm.on_trade_opened()
        self._log(f"ORDER PLACED #{result.order}: {summary}")
        self.logger.log_event(time=self.last_bar_time, event="order_placed", reason="ok",
                              ticket=result.order, risk_money=lot.normalized_risk, **request)
        self.persist_state()

    def _filling_mode(self):
        """Pick a filling mode the symbol actually supports."""
        allowed = self.symbol_info.filling_mode
        if allowed & 1:
            return self.mt5.ORDER_FILLING_FOK
        if allowed & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Everything a dashboard needs to render, per Section 12."""
        with self._lock:
            snap: dict = {
                "status": self.status,
                "mode": "LIVE" if self.live else "DRY_RUN",
                "symbol": self.symbol,
                "risk_percent": self.config.risk_percent_initial_balance,
                "log": list(self.log_lines)[-120:],
                "timestamp": self._now(),
            }
            if self.bars is None:
                return snap

            i = len(self.bars) - 1
            regime = self._regime_at(i)
            snap["last_bar"] = str(self.last_bar_time)
            snap["regime"] = regime.value

            # Six EMA200 regime lights.
            lights = []
            for tf in self.config.regime_timeframes:
                close = self.regime_frame[f"{tf}_close"].iloc[i]
                ema = self.regime_frame[f"{tf}_ema"].iloc[i]
                if pd.isna(close) or pd.isna(ema):
                    side = "UNKNOWN"
                else:
                    side = "ABOVE" if close > ema else ("BELOW" if close < ema else "ON")
                lights.append({
                    "timeframe": tf,
                    "close": None if pd.isna(close) else round(float(close), 2),
                    "ema200": None if pd.isna(ema) else round(float(ema), 2),
                    "side": side,
                })
            snap["regime_lights"] = lights

            wpr = float(self.bars["wpr"].iloc[i])
            atr = float(self.bars["atr"].iloc[i])
            snap["indicators"] = {
                "wpr": round(wpr, 1),
                "atr": round(atr, 2),
                "wpr_state": (
                    "OVERSOLD" if wpr <= self.config.wpr_oversold
                    else "OVERBOUGHT" if wpr >= self.config.wpr_overbought
                    else "NORMAL"
                ),
            }

            # Setup state machine.
            setup_info = {"state": self.sm.state.value}
            if self.sm.setup is not None:
                s = self.sm.setup
                setup_info.update({
                    "direction": s.direction,
                    "zone_type": s.zone.type.value,
                    "zone_low": round(s.zone.low, 2),
                    "zone_high": round(s.zone.high, 2),
                    "confluence": s.confluence_score,
                    "bars_left": max(0, s.expiry_bar - i),
                    "wpr_extreme_seen": s.wpr_extreme_seen,
                    "wpr_confirmed": s.wpr_confirmed,
                    "push1": s.push1_bar is not None,
                    "push2": s.push2_bar is not None,
                })
            snap["setup"] = setup_info

            snap["zones"] = [
                {
                    "id": z.id, "type": z.type.value, "direction": z.direction,
                    "low": round(z.low, 2), "high": round(z.high, 2),
                    "age": i - z.created_bar, "touched": z.touched,
                }
                for z in sorted(self.zone_engine.active_zones, key=lambda z: -z.created_bar)[:25]
            ]

            state = self.safety.state
            peak = state.running_equity_peak or 1.0
            snap["safety"] = {
                "initial_balance": round(state.initial_balance, 2),
                "closed_balance": round(state.closed_balance, 2),
                "equity_peak": round(peak, 2),
                "trades_today": state.trades_today,
                "losses_today": state.losses_today,
                "daily_realized_r": round(state.daily_realized_r, 2),
                "open_positions": state.open_positions,
                "cooldown_until_bar": state.cooldown_until_bar,
                "cooldown_active": i < state.cooldown_until_bar,
                "drawdown_locked": state.drawdown_locked,
                "target_reached": state.target_reached,
                "risk_budget": round(state.initial_balance * self.config.risk_fraction(), 2),
                "max_trades_per_day": self.config.max_trades_per_day,
                "max_losses_per_day": self.config.max_losses_per_day,
            }

        # MT5 calls outside the lock -- they can block on the terminal.
        try:
            account = self.mt5.account_info()
            tick = self.mt5.symbol_info_tick(self.symbol)
            if account:
                equity = account.equity
                snap["account"] = {
                    "login": account.login, "server": account.server,
                    "type": TRADE_MODES.get(account.trade_mode, "UNKNOWN"),
                    "balance": round(account.balance, 2),
                    "equity": round(equity, 2),
                    "currency": account.currency,
                    "drawdown_percent": round(max(0.0, (1 - equity / peak) * 100), 2),
                }
            if tick:
                snap["price"] = {
                    "bid": tick.bid, "ask": tick.ask,
                    "spread": round(tick.ask - tick.bid, 2),
                }
            all_positions = self.mt5.positions_get() or []
            positions = [
                p for p in all_positions if p.magic == MAGIC and p.symbol == self.symbol
            ]
            # Positions this robot does not own still move account equity, and the
            # peak-equity circuit breaker reads account equity -- so manual trades
            # can trip the robot's lock. Surface them rather than hiding them.
            foreign = [p for p in all_positions if p.magic != MAGIC]
            snap["foreign_positions"] = [
                {
                    "ticket": p.ticket, "symbol": p.symbol,
                    "direction": "BUY" if p.type == 0 else "SELL",
                    "volume": p.volume, "profit": round(p.profit, 2),
                }
                for p in foreign
            ]
            snap["position"] = (
                {
                    "ticket": positions[0].ticket,
                    "direction": "BUY" if positions[0].type == 0 else "SELL",
                    "volume": positions[0].volume,
                    "open_price": positions[0].price_open,
                    "sl": positions[0].sl, "tp": positions[0].tp,
                    "profit": round(positions[0].profit, 2),
                }
                if positions else None
            )
        except Exception as exc:
            snap["mt5_error"] = f"{type(exc).__name__}: {exc}"
        return snap

    # ------------------------------------------------------------------
    def set_live(self, live: bool) -> str:
        """Arm or disarm real order placement while the loop runs."""
        if live:
            terminal = self.mt5.terminal_info()
            if terminal is None or not terminal.trade_allowed:
                return "AutoTrading is OFF in the MT5 terminal -- enable the Algo Trading button first."
            account = self.mt5.account_info()
            mode = TRADE_MODES.get(account.trade_mode, "UNKNOWN") if account else "UNKNOWN"
            if mode != "DEMO" and not self.allow_real_money:
                return f"refusing to arm live trading on a {mode} account."
        self.live = live
        self._log(f"mode changed to {'LIVE ORDER PLACEMENT' if live else 'DRY RUN'}")
        return ""

    def close_position(self) -> str:
        """Manually flatten the robot's open position."""
        positions = [
            p for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == MAGIC
        ]
        if not positions:
            return "no open position"
        p = positions[0]
        tick = self.mt5.symbol_info_tick(self.symbol)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": p.volume,
            "type": self.mt5.ORDER_TYPE_SELL if p.type == 0 else self.mt5.ORDER_TYPE_BUY,
            "position": p.ticket,
            "price": tick.bid if p.type == 0 else tick.ask,
            "deviation": 20,
            "magic": MAGIC,
            "comment": "manual close from dashboard",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            message = f"close failed: retcode {getattr(result, 'retcode', None)}"
            self._log(message)
            return message
        self._log(f"position #{p.ticket} closed manually from dashboard")
        return ""

    # ------------------------------------------------------------------
    def run(self, max_bars: Optional[int] = None, stop_event: Optional[threading.Event] = None,
            shutdown_on_exit: bool = True) -> None:
        self._log("live loop started")
        self._log("expect long quiet periods: this strategy averaged ~1.7 trades/month in backtest")
        self.status = "running"
        processed = 0
        try:
            while max_bars is None or processed < max_bars:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    latest = self._fetch_bars(3)
                    newest_closed = latest.index[-2]
                    if newest_closed > self.last_bar_time:
                        with self._lock:
                            self.on_new_bar()
                        processed += 1
                        self._log(f"bar {self.last_bar_time} processed | "
                                  f"regime {self._regime_at(len(self.bars) - 1).value} | "
                                  f"state {self.sm.state.value}")
                except Exception as exc:  # keep the loop alive across transient errors
                    self._log(f"WARNING: {type(exc).__name__}: {exc}")

                # Sleep in slices so a stop request is honoured promptly.
                for _ in range(max(1, self.poll_seconds)):
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            self._log("stopped by user")
        finally:
            self.status = "stopped"
            self.persist_state()
            if self.logger.events or self.logger.trades:
                self.logger.write(self.log_dir)
                self._log(f"session logs written to {self.log_dir}")
            if shutdown_on_exit:
                self.mt5.shutdown()
