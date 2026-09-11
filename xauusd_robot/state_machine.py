"""Section 7: Expert Advisor State Machine.

IDLE -> ZONE_FOUND -> REACTION_ARMED -> WPR_CONFIRMED -> PUSH_1 -> READY
     -> IN_TRADE -> COOLDOWN -> IDLE

Zone discovery and the touch/reaction rules (IDLE -> ZONE_FOUND ->
REACTION_ARMED) live in :mod:`xauusd_robot.zones`; this module owns the
armed setup: the 5-bar expiry timer, WPR(49) extreme/exit sequencing, and
the two push-candle trigger. The state machine exists specifically so
rules cannot be evaluated out of order -- e.g. the robot must never
recognise two push candles first and retroactively discover an old zone.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .candles import Candle, is_qualifying_push, push2_breaks_push1
from .zones import ReactionEvent, Zone


class SetupState(enum.Enum):
    IDLE = "IDLE"
    ZONE_FOUND = "ZONE_FOUND"
    REACTION_ARMED = "REACTION_ARMED"
    WPR_CONFIRMED = "WPR_CONFIRMED"
    PUSH_1 = "PUSH_1"
    READY = "READY"
    IN_TRADE = "IN_TRADE"
    COOLDOWN = "COOLDOWN"


@dataclass
class Setup:
    direction: str
    zone: Zone
    touch_bar: int
    reaction_bar: int
    reaction_time: pd.Timestamp
    expiry_bar: int
    confluence_score: int
    confluence_types: list
    reaction_extreme: float
    state: SetupState = SetupState.REACTION_ARMED
    wpr_extreme_seen: bool = False
    wpr_extreme_bar: Optional[int] = None
    wpr_extreme_value: Optional[float] = None
    wpr_confirmed: bool = False
    wpr_exit_bar: Optional[int] = None
    wpr_exit_value: Optional[float] = None
    push1: Optional[Candle] = None
    push1_bar: Optional[int] = None
    push2: Optional[Candle] = None
    push2_bar: Optional[int] = None


@dataclass
class SetupOutcome:
    kind: str  # "none" | "expired" | "zone_invalidated" | "entry_ready"
    setup: Optional[Setup] = None
    detail: dict = field(default_factory=dict)


class SetupStateMachine:
    def __init__(self, config, bars: pd.DataFrame):
        self.config = config
        self.update_bars(bars)
        self.setup: Optional[Setup] = None
        self.state: SetupState = SetupState.IDLE

    def update_bars(self, bars: pd.DataFrame) -> None:
        """Refresh cached arrays without disturbing the armed setup.

        Used by the live loop, which appends one closed bar at a time. Bars
        may only be appended: an armed setup holds integer bar indices.
        """
        self.index = bars.index
        self.open = bars["open"].to_numpy()
        self.high = bars["high"].to_numpy()
        self.low = bars["low"].to_numpy()
        self.close = bars["close"].to_numpy()
        self.wpr = bars["wpr"].to_numpy()
        self.macd_hist = (bars["macd_hist"].to_numpy()
                          if "macd_hist" in bars.columns else None)

    # ------------------------------------------------------------------
    @property
    def is_idle(self) -> bool:
        return self.state == SetupState.IDLE

    def candle(self, i: int) -> Candle:
        return Candle(self.open[i], self.high[i], self.low[i], self.close[i])

    # ------------------------------------------------------------------
    def arm(self, event: ReactionEvent) -> Setup:
        """Transition REACTION_ARMED and back-fill WPR / push-1 state.

        The reaction candle itself may already satisfy the WPR sequencing
        and may simultaneously count as Push Candle 1 (Section 4.6).
        """
        cfg = self.config
        i = event.reaction_bar
        lo, hi = event.touch_bar, i
        if event.direction == "BUY":
            extreme = float(np.min(self.low[lo : hi + 1]))
        else:
            extreme = float(np.max(self.high[lo : hi + 1]))

        setup = Setup(
            direction=event.direction,
            zone=event.zone,
            touch_bar=event.touch_bar,
            reaction_bar=i,
            reaction_time=event.reaction_time,
            expiry_bar=i + cfg.setup_expiry_bars,
            confluence_score=event.confluence_score,
            confluence_types=event.confluence_types,
            reaction_extreme=extreme,
        )
        self.setup = setup
        self.state = SetupState.REACTION_ARMED

        # WPR extreme may begin no earlier than wpr_max_lead_bars before the
        # first zone touch; scan that window up to and including the
        # reaction bar, in order.
        start = max(0, event.touch_bar - cfg.wpr_max_lead_bars)
        for b in range(start, i + 1):
            self._advance_wpr(setup, b)
        self._advance_push(setup, i)
        self._sync_state(setup)
        return setup

    # ------------------------------------------------------------------
    def update(self, i: int, regime) -> SetupOutcome:
        """Advance the armed setup with newly closed bar ``i``."""
        setup = self.setup
        if setup is None or self.state in (SetupState.IDLE, SetupState.IN_TRADE, SetupState.COOLDOWN):
            return SetupOutcome("none")

        # Cancel if the originating zone invalidates (close beyond far boundary).
        if (setup.direction == "BUY" and self.close[i] < setup.zone.far_boundary) or (
            setup.direction == "SELL" and self.close[i] > setup.zone.far_boundary
        ):
            self.cancel()
            return SetupOutcome("zone_invalidated", setup)

        if i > setup.expiry_bar:
            self.cancel()
            return SetupOutcome("expired", setup)

        # Regime must remain aligned with the setup direction; on a MIXED or
        # opposite regime bar the setup is frozen (pseudocode returns early)
        # and will simply age out if alignment does not return.
        if getattr(regime, "value", regime) != setup.direction:
            return SetupOutcome("none", setup)

        if setup.direction == "BUY":
            setup.reaction_extreme = min(setup.reaction_extreme, float(self.low[i]))
        else:
            setup.reaction_extreme = max(setup.reaction_extreme, float(self.high[i]))

        self._advance_wpr(setup, i)
        entry_ready = self._advance_push(setup, i)
        self._sync_state(setup)

        if entry_ready and setup.wpr_confirmed:
            setup.state = SetupState.READY
            self.state = SetupState.READY
            return SetupOutcome("entry_ready", setup, {"entry_bar": i, "entry_price": float(self.close[i])})
        return SetupOutcome("none", setup)

    # ------------------------------------------------------------------
    def _advance_wpr(self, setup: Setup, i: int) -> None:
        """Advance the momentum gate. Field names are historical -- they hold the
        generic primed/confirmed pair whichever filter is selected."""
        cfg = self.config
        if cfg.momentum_filter == "none":
            setup.wpr_extreme_seen = True
            setup.wpr_confirmed = True
            return
        if cfg.momentum_filter == "macd":
            self._advance_macd(setup, i)
            return

        value = self.wpr[i]
        if np.isnan(value):
            return
        if setup.direction == "BUY":
            if value <= cfg.wpr_oversold:
                setup.wpr_extreme_seen = True
                setup.wpr_extreme_bar = i
                setup.wpr_extreme_value = float(value)
            elif setup.wpr_extreme_seen and not setup.wpr_confirmed and value > cfg.wpr_oversold:
                setup.wpr_confirmed = True
                setup.wpr_exit_bar = i
                setup.wpr_exit_value = float(value)
        else:
            if value >= cfg.wpr_overbought:
                setup.wpr_extreme_seen = True
                setup.wpr_extreme_bar = i
                setup.wpr_extreme_value = float(value)
            elif setup.wpr_extreme_seen and not setup.wpr_confirmed and value < cfg.wpr_overbought:
                setup.wpr_confirmed = True
                setup.wpr_exit_bar = i
                setup.wpr_exit_value = float(value)

    def _advance_macd(self, setup: Setup, i: int) -> None:
        """MACD analogue of the WPR rule: histogram against the trade (primed),
        then crossing through zero in the trade's favour (confirmed)."""
        if self.macd_hist is None:
            return
        value = self.macd_hist[i]
        if np.isnan(value):
            return
        if setup.direction == "BUY":
            if value < 0:
                setup.wpr_extreme_seen = True
            elif setup.wpr_extreme_seen and not setup.wpr_confirmed and value > 0:
                setup.wpr_confirmed = True
        else:
            if value > 0:
                setup.wpr_extreme_seen = True
            elif setup.wpr_extreme_seen and not setup.wpr_confirmed and value < 0:
                setup.wpr_confirmed = True

    def _advance_push(self, setup: Setup, i: int) -> bool:
        """Track the two-consecutive-push-candle chain. Returns True on Push 2."""
        cfg = self.config
        if not cfg.require_push_candles:
            # Trigger removed: the WPR exit alone is the entry signal.
            return True

        # Push 2 describes THIS bar completing the chain, nothing more. Without
        # clearing it each bar it survived a broken chain: once set, a blocked
        # entry (WPR not yet confirmed) left it latched while Push 1 reset
        # underneath, so the dashboard showed Push 2 satisfied with Push 1 dark
        # and the funnel counted chain completions that could never trade.
        setup.push2, setup.push2_bar = None, None

        cand = self.candle(i)
        p1_threshold = cfg.push_body_ratio
        p2_threshold = cfg.push2_body_ratio if cfg.push2_body_ratio is not None else p1_threshold
        # The same candle is judged by a different bar depending on which role it
        # is being considered for.
        as_push1 = is_qualifying_push(cand, setup.direction, p1_threshold)
        as_push2 = is_qualifying_push(cand, setup.direction, p2_threshold)

        if cfg.push_candles_required <= 1:
            # One push candle: keep the directional body-quality test, drop the
            # continuation confirmation that Push 2 provides.
            if as_push1:
                setup.push1, setup.push1_bar = cand, i
                return True
            setup.push1, setup.push1_bar = None, None
            return False

        if setup.push1 is not None and setup.push1_bar == i - 1:
            if as_push2 and (
                not cfg.push2_breaks_push1 or push2_breaks_push1(setup.push1, cand, setup.direction)
            ):
                setup.push2 = cand
                setup.push2_bar = i
                return True
            # Failed as Push 2, but it may still be strong enough to start a
            # fresh chain as Push 1.
            setup.push1, setup.push1_bar = (cand, i) if as_push1 else (None, None)
            return False

        setup.push1, setup.push1_bar = (cand, i) if as_push1 else (None, None)
        return False

    def _sync_state(self, setup: Setup) -> None:
        if setup.push1 is not None and setup.wpr_confirmed:
            new_state = SetupState.PUSH_1
        elif setup.wpr_confirmed:
            new_state = SetupState.WPR_CONFIRMED
        else:
            new_state = SetupState.REACTION_ARMED
        setup.state = new_state
        self.state = new_state

    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self.setup = None
        self.state = SetupState.IDLE

    def on_trade_opened(self) -> None:
        self.state = SetupState.IN_TRADE
        if self.setup is not None:
            self.setup.state = SetupState.IN_TRADE

    def on_trade_closed(self) -> None:
        self.setup = None
        self.state = SetupState.IDLE
