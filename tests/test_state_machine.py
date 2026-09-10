"""Sections 3.3 / 3.4 / 4.5 / 4.6 / 7: WPR sequencing, push candles, setup expiry."""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_bars
from xauusd_robot.regime import Regime
from xauusd_robot.state_machine import SetupState, SetupStateMachine
from xauusd_robot.zones import ReactionEvent, Zone, ZoneType


def make_zone(direction="BUY", low=100.0, high=101.0, created_bar=0):
    return Zone(
        id=1, type=ZoneType.FVG, direction=direction, low=low, high=high,
        created_bar=created_bar, created_time=pd.Timestamp("2024-01-01"), expires_bar=created_bar + 96,
    )


def make_event(zone, touch_bar, reaction_bar, index):
    return ReactionEvent(
        zone=zone, direction=zone.direction, reaction_bar=reaction_bar,
        reaction_time=index[reaction_bar], touch_bar=touch_bar,
        confluence_score=1, confluence_types=[zone.type.value],
    )


def build(rows, wpr):
    bars = make_bars(rows, wpr=wpr)
    return bars, SetupStateMachine(__import__("xauusd_robot.config", fromlist=["x"]).StrategyConfig(), bars)


# ----------------------------------------------------------------- WPR ----
def test_buy_setup_requires_wpr_extreme_then_exit_above_minus_eighty():
    rows = [
        (100.5, 100.6, 100.0, 100.2),  # 0 touch
        (100.2, 101.3, 100.1, 101.2),  # 1 reaction close above 101.0
        (101.2, 102.0, 101.1, 101.9),  # 2
        (101.9, 103.0, 101.8, 102.9),  # 3
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0, -55.0])
    setup = sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert setup.wpr_extreme_seen and setup.wpr_confirmed
    assert setup.wpr_extreme_value == pytest.approx(-85.0)
    assert setup.wpr_exit_value == pytest.approx(-70.0)


def test_wpr_extreme_older_than_max_lead_bars_does_not_qualify():
    """The -85 print is 4 bars before the touch, outside the 2-bar lead window."""
    rows = [
        (100.5, 100.6, 100.0, 100.2),  # 0  <- extreme here
        (100.2, 100.7, 100.1, 100.6),  # 1
        (100.6, 100.8, 100.2, 100.7),  # 2
        (100.7, 100.9, 100.3, 100.8),  # 3
        (100.8, 100.9, 100.0, 100.4),  # 4 touch
        (100.4, 101.3, 100.3, 101.2),  # 5 reaction
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0, -55.0, -50.0, -45.0])
    setup = sm.arm(make_event(make_zone(), touch_bar=4, reaction_bar=5, index=bars.index))
    assert not setup.wpr_extreme_seen
    assert not setup.wpr_confirmed


def test_wpr_extreme_two_bars_before_touch_is_still_eligible():
    rows = [
        (100.5, 100.6, 100.0, 100.2),  # 0
        (100.2, 100.7, 100.1, 100.6),  # 1
        (100.6, 100.9, 100.0, 100.4),  # 2 touch
        (100.4, 101.3, 100.3, 101.2),  # 3 reaction
    ]
    bars, sm = build(rows, wpr=[-85.0, -75.0, -60.0, -55.0])
    setup = sm.arm(make_event(make_zone(), touch_bar=2, reaction_bar=3, index=bars.index))
    assert setup.wpr_extreme_seen and setup.wpr_confirmed


def test_sell_setup_requires_wpr_above_minus_twenty_then_close_below():
    rows = [
        (101.0, 101.5, 100.9, 101.2),
        (101.2, 101.3, 100.0, 100.1),
    ]
    bars, sm = build(rows, wpr=[-10.0, -35.0])
    zone = make_zone(direction="SELL", low=101.0, high=102.0)
    setup = sm.arm(make_event(zone, touch_bar=0, reaction_bar=1, index=bars.index))
    assert setup.wpr_extreme_seen and setup.wpr_confirmed


# ---------------------------------------------------------- push candles ----
def test_two_consecutive_push_candles_trigger_entry():
    rows = [
        (100.5, 100.6, 100.0, 100.2),   # 0 touch
        (100.2, 101.2, 100.15, 101.1),  # 1 reaction, body/range = 0.86 -> also Push 1
        (101.1, 101.9, 101.05, 101.85), # 2 Push 2, closes above Push 1 high (101.2)
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.setup.push1_bar == 1  # the reaction candle counted as Push 1

    outcome = sm.update(2, Regime.BUY)
    assert outcome.kind == "entry_ready"
    assert sm.state is SetupState.READY
    assert outcome.detail["entry_price"] == pytest.approx(101.85)


def test_push2_must_close_beyond_push1_high():
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.15, 101.1),   # Push 1, high 101.2
        (101.1, 101.19, 101.0, 101.17),  # strong body but closes under 101.2
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.update(2, Regime.BUY).kind == "none"


def test_weak_body_candle_breaks_the_push_chain():
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.15, 101.1),   # Push 1
        (101.1, 102.0, 101.0, 101.4),    # body/range = 0.30 -> not a push candle
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.update(2, Regime.BUY).kind == "none"
    assert sm.setup.push1 is None


def test_push_candles_must_be_consecutive():
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.15, 101.1),   # Push 1 at bar 1
        (101.1, 101.15, 101.0, 101.05),  # doji-ish gap bar, chain resets
        (101.05, 102.0, 101.0, 101.95),  # strong, but Push 1 is gone
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0, -55.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.update(2, Regime.BUY).kind == "none"
    assert sm.update(3, Regime.BUY).kind == "none"  # bar 3 becomes the new Push 1 only


# ------------------------------------------------------------- lifetime ----
def test_setup_expires_after_five_post_reaction_bars():
    rows = [(100.5, 100.6, 100.0, 100.2), (100.2, 101.2, 100.1, 101.1)] + [
        (101.1, 101.2, 101.0, 101.05)
    ] * 8
    bars, sm = build(rows, wpr=[-85.0, -70.0] + [-60.0] * 8)
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.setup.expiry_bar == 6

    for i in range(2, 7):
        assert sm.update(i, Regime.BUY).kind == "none"
    assert sm.update(7, Regime.BUY).kind == "expired"
    assert sm.state is SetupState.IDLE


def test_setup_cancels_when_zone_invalidates():
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.1, 101.1),
        (101.1, 101.2, 99.0, 99.5),  # closes below the zone's far boundary (100.0)
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.update(2, Regime.BUY).kind == "zone_invalidated"
    assert sm.state is SetupState.IDLE


def test_setup_does_not_advance_on_a_mixed_regime_bar():
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.15, 101.1),
        (101.1, 101.9, 101.05, 101.85),  # would be Push 2
    ]
    bars, sm = build(rows, wpr=[-85.0, -70.0, -60.0])
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert sm.update(2, Regime.MIXED).kind == "none"


def test_entry_blocked_until_wpr_confirms():
    """Push 1 + Push 2 alone are not enough while WPR is still in the extreme."""
    rows = [
        (100.5, 100.6, 100.0, 100.2),
        (100.2, 101.2, 100.15, 101.1),
        (101.1, 101.9, 101.05, 101.85),
    ]
    bars, sm = build(rows, wpr=[-85.0, -90.0, -95.0])  # never exits the extreme
    sm.arm(make_event(make_zone(), touch_bar=0, reaction_bar=1, index=bars.index))
    assert not sm.setup.wpr_confirmed
    assert sm.update(2, Regime.BUY).kind == "none"
