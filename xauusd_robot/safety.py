"""Section 5.5 / 15 / 12: circuit breakers, daily limits, and restart persistence.

``SafetyState`` is a small, JSON-serializable dataclass so it can be
persisted to disk and reloaded after an EA/terminal restart without
losing the initial-balance reference, the running equity peak, or any
lock state (Acceptance Test: "Restart").
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional, Tuple


@dataclass
class SafetyState:
    initial_balance: float
    closed_balance: float
    running_equity_peak: float
    broker_day: Optional[str] = None
    trades_today: int = 0
    losses_today: int = 0
    daily_realized_r: float = 0.0
    open_positions: int = 0
    cooldown_until_bar: int = -1
    drawdown_locked: bool = False
    target_reached: bool = False

    @classmethod
    def new(cls, initial_balance: float) -> "SafetyState":
        return cls(
            initial_balance=initial_balance,
            closed_balance=initial_balance,
            running_equity_peak=initial_balance,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "SafetyState":
        return cls(**json.loads(text))


class SafetyEngine:
    """Stateless rule evaluator operating on a mutable :class:`SafetyState`."""

    def __init__(self, config, state: SafetyState):
        self.config = config
        self.state = state

    # ------------------------------------------------------------------
    def roll_broker_day(self, current_day: date) -> None:
        day_str = current_day.isoformat()
        if self.state.broker_day != day_str:
            self.state.broker_day = day_str
            self.state.trades_today = 0
            self.state.losses_today = 0
            self.state.daily_realized_r = 0.0

    def update_equity(self, current_equity: float) -> None:
        cfg = self.config
        if current_equity > self.state.running_equity_peak:
            self.state.running_equity_peak = current_equity
        if self.state.running_equity_peak > 0:
            drawdown = 1.0 - current_equity / self.state.running_equity_peak
            if drawdown >= cfg.max_peak_equity_drawdown:
                self.state.drawdown_locked = True

    def register_trade_open(self) -> None:
        self.state.open_positions += 1

    def register_trade_close(self, realized_r: float, closed_balance: float, current_bar_index: int) -> None:
        cfg = self.config
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.closed_balance = closed_balance
        self.state.trades_today += 1
        self.state.daily_realized_r += realized_r
        if realized_r < 0:
            self.state.losses_today += 1
        self.state.cooldown_until_bar = current_bar_index + cfg.cooldown_bars
        if closed_balance >= cfg.target_multiplier * self.state.initial_balance:
            self.state.target_reached = True
        if closed_balance > self.state.running_equity_peak:
            self.state.running_equity_peak = closed_balance

    def manual_reset_drawdown_lock(self) -> None:
        self.state.drawdown_locked = False
        self.state.running_equity_peak = self.state.closed_balance

    def manual_reset_target(self) -> None:
        self.state.target_reached = False

    # ------------------------------------------------------------------
    def can_open_new_trade(
        self,
        current_bar_index: int,
        spread: float,
        sl_distance: float,
        atr: float,
    ) -> Tuple[bool, str]:
        cfg = self.config
        s = self.state
        if s.target_reached:
            return False, "target_reached"
        if s.drawdown_locked:
            return False, "drawdown_lock"
        if s.open_positions >= cfg.max_open_positions:
            return False, "position_open"
        if s.trades_today >= cfg.max_trades_per_day:
            return False, "max_trades_per_day"
        if s.losses_today >= cfg.max_losses_per_day:
            return False, "max_losses_per_day"
        if s.daily_realized_r <= cfg.daily_loss_limit_r:
            return False, "daily_loss_limit"
        if current_bar_index < s.cooldown_until_bar:
            return False, "cooldown"
        if sl_distance > 0 and spread > cfg.max_spread_vs_sl * sl_distance:
            return False, "spread_vs_sl"
        if atr and atr > 0 and spread > cfg.max_spread_atr * atr:
            return False, "spread_vs_atr"
        return True, "ok"
