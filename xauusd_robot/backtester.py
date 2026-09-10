"""Bar-by-bar backtest harness implementing the Section 3.5 entry sequence
and the compact pseudocode at the end of the blueprint.

Execution model
---------------
Input prices are treated as BID. A BUY fills at ask (bid + spread) and
exits at bid; a SELL fills at bid and exits at ask. The spread is
therefore charged exactly once per round trip, and the structural stop
distance used for sizing is measured from the real fill price to the real
stop level -- so the configured risk budget is the true worst-case loss,
not an idealised mid-price one.

Conservative assumptions (documented deliberately):
* If a bar's range contains both the stop and the target, the STOP is
  assumed to fill first.
* A bar that opens beyond a level fills at the open (gap fill), which is
  worse than the level for stops and better for targets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import build_regime_frame, build_timeframe_indicators
from .indicators import add_core_indicators
from .logging_engine import RunLogger
from .regime import Regime, compute_regime
from .risk import build_trade_plan
from .safety import SafetyEngine, SafetyState
from .state_machine import SetupStateMachine
from .zones import ZoneEngine


@dataclass
class OpenPosition:
    direction: str
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_trigger: float    # bid-space level the simulator compares bars against
    target_trigger: float  # bid-space level the simulator compares bars against
    stop_price: float      # the order level as the broker would show it
    target_price: float    # the order level as the broker would show it
    stop_distance: float
    lots: float
    risk_money: float
    spread: float
    zone_type: str
    zone_id: int
    confluence_score: int
    monetary_loss_per_lot: float
    #: Best favourable excursion reached, in R. Bars where the stop was hit
    #: are excluded, matching the simulator's stop-before-target assumption,
    #: so this never flatters a laddered-exit analysis.
    mfe_r: float = 0.0


@dataclass
class BacktestResult:
    config: StrategyConfig
    initial_balance: float
    final_balance: float
    trades: pd.DataFrame
    events: pd.DataFrame
    equity: pd.DataFrame
    metrics: Dict = field(default_factory=dict)
    safety_state: Optional[SafetyState] = None


class Backtester:
    def __init__(
        self,
        m5: pd.DataFrame,
        config: StrategyConfig = StrategyConfig(),
        initial_balance: float = 1000.0,
        safety_state: Optional[SafetyState] = None,
    ):
        self.raw = m5
        self.config = config
        self.initial_balance = initial_balance
        self.logger = RunLogger()
        self.state = safety_state or SafetyState.new(initial_balance)
        self.safety = SafetyEngine(config, self.state)
        self.position: Optional[OpenPosition] = None
        self.balance = self.state.closed_balance
        # Section 14: the funnel makes "why did it not act" answerable at a
        # glance, before anyone digs into the per-setup event log.
        self.funnel: Counter = Counter()
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        cfg = self.config
        self.bars = add_core_indicators(self.raw, cfg.ema_period, cfg.atr_period, cfg.wpr_period)
        tf_frames = build_timeframe_indicators(self.raw, cfg)
        regime_frame = build_regime_frame(self.bars, tf_frames, cfg)
        self.regime = compute_regime(regime_frame, cfg.regime_timeframes)
        self.regime_frame = regime_frame

        tick = cfg.broker.tick_size
        if "spread" in self.raw.columns:
            spread_price = self.raw["spread"].astype(float) * tick
        else:
            spread_price = pd.Series(cfg.broker.default_spread_points * tick, index=self.raw.index)
        self.spread = spread_price.to_numpy()

        self.zone_engine = ZoneEngine(cfg, self.bars)
        self.sm = SetupStateMachine(cfg, self.bars)

        self.open_ = self.bars["open"].to_numpy()
        self.high = self.bars["high"].to_numpy()
        self.low = self.bars["low"].to_numpy()
        self.close = self.bars["close"].to_numpy()
        self.atr = self.bars["atr"].to_numpy()
        self.wpr = self.bars["wpr"].to_numpy()
        self.index = self.bars.index
        self.regime_values = self.regime.to_numpy()

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        n = len(self.bars)
        for i in range(n):
            ts = self.index[i]
            self.funnel["bars"] += 1
            self.safety.roll_broker_day(ts.date())

            if self.position is not None and i > self.position.entry_bar:
                self._manage_position(i)

            equity = self._current_equity(i)
            self.safety.update_equity(equity)
            self.logger.log_equity(
                time=ts, balance=self.balance, equity=equity,
                peak=self.state.running_equity_peak,
                drawdown=1.0 - equity / self.state.running_equity_peak if self.state.running_equity_peak else 0.0,
                regime=self._regime_at(i).value, state=self.sm.state.value,
            )

            if self.state.target_reached or self.state.drawdown_locked:
                # Locked: pending setups cancel, no new entries until manual reset.
                if self.sm.setup is not None and self.position is None:
                    self.sm.cancel()
                continue

            events = self.zone_engine.process_bar(i)
            regime = self._regime_at(i)
            self.funnel[f"regime_{regime.value}"] += 1
            self.funnel["zone_reactions"] += len(events)
            self.funnel["zone_reactions_regime_matched"] += sum(
                1 for e in events if e.direction == regime.value
            )

            if self.sm.setup is not None:
                outcome = self.sm.update(i, regime)
                if outcome.kind in ("expired", "zone_invalidated"):
                    self._log_setup_end(i, outcome)
                elif outcome.kind == "entry_ready":
                    self._attempt_entry(i, outcome)

            if self.sm.is_idle and self.position is None and regime is not Regime.MIXED:
                if i >= self.state.cooldown_until_bar:
                    for event in events:
                        if event.direction == regime.value:
                            self.sm.arm(event)
                            self.funnel["setups_armed"] += 1
                            self._log_arm(i, event)
                            break

        if self.position is not None:
            self._close_position(len(self.bars) - 1, float(self.close[-1]), "end_of_data")

        return self._build_result()

    # ------------------------------------------------------------------
    def _regime_at(self, i: int) -> Regime:
        value = self.regime_values[i]
        return value if isinstance(value, Regime) else Regime(value)

    def _current_equity(self, i: int) -> float:
        if self.position is None:
            return self.balance
        return self.balance + self._floating_pnl(i)

    def _floating_pnl(self, i: int) -> float:
        p = self.position
        if p is None:
            return 0.0
        if p.direction == "BUY":
            move = float(self.close[i]) - p.entry_price
        else:
            move = p.entry_price - (float(self.close[i]) + p.spread)
        return self._price_to_money(move, p.lots)

    def _price_to_money(self, price_move: float, lots: float) -> float:
        b = self.config.broker
        return price_move / b.tick_size * b.tick_value * lots

    # ------------------------------------------------------------------
    def _favourable_r(self, p: "OpenPosition", bar_high: float, bar_low: float) -> float:
        """How far this bar ran in the trade's favour, in R."""
        if p.stop_distance <= 0:
            return 0.0
        move = (bar_high - p.entry_price) if p.direction == "BUY" else (p.entry_price - bar_low)
        return move / p.stop_distance

    def _manage_position(self, i: int) -> None:
        p = self.position
        bar_open, bar_high, bar_low = float(self.open_[i]), float(self.high[i]), float(self.low[i])

        if p.direction == "BUY":
            hit_stop = bar_low <= p.stop_trigger
            hit_target = bar_high >= p.target_trigger
            if not hit_stop:
                p.mfe_r = max(p.mfe_r, self._favourable_r(p, bar_high, bar_low))
            if hit_stop:
                exit_price = bar_open if bar_open <= p.stop_trigger else p.stop_trigger
                self._close_position(i, exit_price, "stop_loss")
                return
            if hit_target:
                exit_price = bar_open if bar_open >= p.target_trigger else p.target_trigger
                self._close_position(i, exit_price, "take_profit")
                return
        else:
            hit_stop = bar_high >= p.stop_trigger
            hit_target = bar_low <= p.target_trigger
            if not hit_stop:
                p.mfe_r = max(p.mfe_r, self._favourable_r(p, bar_high, bar_low))
            if hit_stop:
                exit_price = bar_open if bar_open >= p.stop_trigger else p.stop_trigger
                self._close_position(i, exit_price, "stop_loss")
                return
            if hit_target:
                exit_price = bar_open if bar_open <= p.target_trigger else p.target_trigger
                self._close_position(i, exit_price, "take_profit")
                return

    def _close_position(self, i: int, exit_bid: float, reason: str) -> None:
        p = self.position
        if p.direction == "BUY":
            pnl_price = exit_bid - p.entry_price
        else:
            pnl_price = p.entry_price - (exit_bid + p.spread)
        money = self._price_to_money(pnl_price, p.lots)
        r_multiple = pnl_price / p.stop_distance if p.stop_distance else 0.0
        if reason == "take_profit":
            p.mfe_r = max(p.mfe_r, r_multiple)

        self.balance += money
        self.safety.register_trade_close(r_multiple, self.balance, i)
        self.position = None
        self.sm.on_trade_closed()

        self.logger.log_trade(
            entry_time=p.entry_time,
            exit_time=self.index[i],
            direction=p.direction,
            zone_type=p.zone_type,
            zone_id=p.zone_id,
            confluence_score=p.confluence_score,
            mfe_r=round(p.mfe_r, 4),
            entry_price=p.entry_price,
            exit_price=exit_bid,
            stop_price=p.stop_price,
            target_price=p.target_price,
            stop_distance=p.stop_distance,
            lots=p.lots,
            spread=p.spread,
            risk_money=p.risk_money,
            pnl_money=money,
            pnl_percent=money / self.initial_balance * 100.0,
            r_multiple=r_multiple,
            exit_reason=reason,
            balance_after=self.balance,
            bars_held=i - p.entry_bar,
            session=_session_of(self.index[i]),
        )
        self.logger.log_event(
            time=self.index[i], event="trade_closed", direction=p.direction,
            reason=reason, r_multiple=r_multiple, pnl_money=money, balance=self.balance,
        )

    # ------------------------------------------------------------------
    def _attempt_entry(self, i: int, outcome) -> None:
        setup = outcome.setup
        self._count_setup_progress(setup)
        self.funnel["setups_entry_evaluated"] += 1
        cfg = self.config
        spread = float(self.spread[i])
        bid_close = float(self.close[i])
        entry_price = bid_close + spread if setup.direction == "BUY" else bid_close
        atr_value = float(self.atr[i])

        plan = build_trade_plan(
            direction=setup.direction,
            entry=entry_price,
            zone_far_boundary=setup.zone.far_boundary,
            reaction_extreme=setup.reaction_extreme,
            atr=atr_value,
            spread=spread,
            initial_balance=self.initial_balance,
            config=cfg,
        )
        # A SELL is entered at bid and exited at ask, so its order levels sit
        # one spread above the bid levels the simulator tests bars against.
        if setup.direction == "SELL":
            stop_trigger = plan.stop_loss.stop_price
            stop_price = stop_trigger + spread
            stop_distance = stop_price - entry_price
            target_price = entry_price - cfg.reward_risk * stop_distance
            target_trigger = target_price - spread
        else:
            stop_trigger = stop_price = plan.stop_loss.stop_price
            stop_distance = entry_price - stop_price
            target_trigger = target_price = entry_price + cfg.reward_risk * stop_distance

        snapshot = self._snapshot(i, setup, entry_price, stop_price, target_price, stop_distance, spread, plan)

        if stop_distance <= 0:
            self._reject(i, setup, "invalid_stop_distance", snapshot)
            return

        if cfg.use_session_filter and _session_of(self.index[i]) not in cfg.allowed_sessions:
            self._reject(i, setup, "session_filter", snapshot)
            return

        allowed, reason = self.safety.can_open_new_trade(i, spread, stop_distance, atr_value)
        if not allowed:
            self._reject(i, setup, reason, snapshot)
            return

        from .risk import compute_lot_size

        risk_budget = self.initial_balance * cfg.risk_fraction()
        lot = compute_lot_size(stop_distance, risk_budget, cfg.broker)
        if not lot.accepted:
            self._reject(i, setup, lot.reason, {**snapshot, "raw_lots": lot.raw_lots})
            return

        # Notional must be expressed in the account currency. Using the symbol's
        # own price is only right when the quote currency IS the account
        # currency; margin_base_rate overrides it for USD-base pairs and crosses.
        base_rate = cfg.broker.margin_base_rate
        if base_rate is None:
            base_rate = entry_price
        required_margin = lot.normalized_lots * cfg.broker.contract_size * base_rate / cfg.broker.leverage
        equity = self._current_equity(i)
        if required_margin > equity:
            self._reject(i, setup, "insufficient_margin", {**snapshot, "required_margin": required_margin})
            return

        self.position = OpenPosition(
            direction=setup.direction,
            entry_bar=i,
            entry_time=self.index[i],
            entry_price=entry_price,
            stop_trigger=stop_trigger,
            target_trigger=target_trigger,
            stop_price=stop_price,
            target_price=target_price,
            stop_distance=stop_distance,
            lots=lot.normalized_lots,
            risk_money=lot.normalized_risk,
            spread=spread,
            zone_type=setup.zone.type.value,
            zone_id=setup.zone.id,
            confluence_score=setup.confluence_score,
            monetary_loss_per_lot=lot.monetary_loss_per_lot,
        )
        self.safety.register_trade_open()
        self.sm.on_trade_opened()
        self.funnel["orders_placed"] += 1
        self.logger.log_event(
            **{**snapshot, "event": "order_placed", "reason": "ok",
               "lots": lot.normalized_lots, "raw_lots": lot.raw_lots,
               "risk_money": lot.normalized_risk, "required_margin": required_margin}
        )

    def _reject(self, i: int, setup, reason: str, snapshot: Dict) -> None:
        self.funnel[f"rejected_{reason}"] += 1
        self.logger.log_event(**{**snapshot, "event": "entry_rejected", "reason": reason})
        self.sm.cancel()

    def _snapshot(self, i, setup, entry, stop, target, stop_distance, spread, plan) -> Dict:
        cfg = self.config
        row = {
            "time": self.index[i],
            "direction": setup.direction,
            "regime": self._regime_at(i).value,
            "zone_id": setup.zone.id,
            "zone_type": setup.zone.type.value,
            "zone_low": setup.zone.low,
            "zone_high": setup.zone.high,
            "zone_age_bars": i - setup.zone.created_bar,
            "confluence_score": setup.confluence_score,
            "confluence_types": "|".join(setup.confluence_types),
            "touch_bar": setup.touch_bar,
            "reaction_bar": setup.reaction_bar,
            "reaction_extreme": setup.reaction_extreme,
            "wpr_extreme_value": setup.wpr_extreme_value,
            "wpr_exit_value": setup.wpr_exit_value,
            "push1_bar": setup.push1_bar,
            "push1_body_ratio": setup.push1.body_range_ratio if setup.push1 else None,
            "push2_bar": setup.push2_bar,
            "push2_body_ratio": setup.push2.body_range_ratio if setup.push2 else None,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "stop_distance": stop_distance,
            "atr": float(self.atr[i]),
            "spread": spread,
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "equity": self._current_equity(i),
            "risk_percent": cfg.risk_percent_initial_balance,
            "risk_budget": self.initial_balance * cfg.risk_fraction(),
            "trades_today": self.state.trades_today,
            "losses_today": self.state.losses_today,
            "daily_realized_r": self.state.daily_realized_r,
            "session": _session_of(self.index[i]),
        }
        for tf in cfg.regime_timeframes:
            row[f"{tf}_close"] = self.regime_frame[f"{tf}_close"].iloc[i]
            row[f"{tf}_ema"] = self.regime_frame[f"{tf}_ema"].iloc[i]
        return row

    def _log_arm(self, i: int, event) -> None:
        self.logger.log_event(
            time=self.index[i], event="setup_armed", reason="reaction_confirmed",
            direction=event.direction, zone_id=event.zone.id, zone_type=event.zone.type.value,
            zone_low=event.zone.low, zone_high=event.zone.high,
            confluence_score=event.confluence_score,
            confluence_types="|".join(event.confluence_types),
            touch_bar=event.touch_bar, reaction_bar=event.reaction_bar,
            regime=self._regime_at(i).value,
        )

    def _log_setup_end(self, i: int, outcome) -> None:
        setup = outcome.setup
        self._count_setup_progress(setup)
        self.funnel[f"setup_{outcome.kind}"] += 1
        self.logger.log_event(
            time=self.index[i], event="setup_cancelled", reason=outcome.kind,
            direction=setup.direction, zone_id=setup.zone.id, zone_type=setup.zone.type.value,
            wpr_extreme_seen=setup.wpr_extreme_seen, wpr_confirmed=setup.wpr_confirmed,
            push1_bar=setup.push1_bar, reaction_bar=setup.reaction_bar,
            regime=self._regime_at(i).value,
        )

    def _count_setup_progress(self, setup) -> None:
        """Record how far a finished setup got, for the funnel report."""
        if setup.wpr_extreme_seen:
            self.funnel["setups_reached_wpr_extreme"] += 1
        if setup.wpr_confirmed:
            self.funnel["setups_reached_wpr_confirmed"] += 1
        if setup.push1 is not None or setup.push2 is not None:
            self.funnel["setups_reached_push1"] += 1
        if setup.push2 is not None:
            self.funnel["setups_reached_push2"] += 1

    # ------------------------------------------------------------------
    def _build_result(self) -> BacktestResult:
        from .metrics import compute_metrics

        for zone in self.zone_engine.all_zones:
            self.funnel[f"zones_created_{zone.type.value}"] += 1
            if zone.retired_reason:
                self.funnel[f"zones_{zone.retired_reason}"] += 1

        trades = self.logger.trades_df()
        equity = self.logger.equity_df()
        metrics = compute_metrics(trades, equity, self.initial_balance, self.config)
        metrics["funnel"] = dict(self.funnel)
        return BacktestResult(
            config=self.config,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            trades=trades,
            events=self.logger.events_df(),
            equity=equity,
            metrics=metrics,
            safety_state=self.state,
        )


def _session_of(ts: pd.Timestamp) -> str:
    """Coarse session label (broker/server time) for Section 9.3 analytics."""
    hour = ts.hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "LATE"
