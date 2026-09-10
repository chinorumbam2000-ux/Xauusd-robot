"""Acceptance tests for Section 4 zone rules (Section 15 table)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import make_bars
from xauusd_robot.zones import ZoneEngine, ZoneType


def run_engine(bars, config, upto=None):
    engine = ZoneEngine(config, bars)
    events = []
    for i in range(upto if upto is not None else len(bars)):
        events.extend(engine.process_bar(i))
    return engine, events


# ---------------------------------------------------------------- FVG ----
def test_bullish_fvg_boundaries_and_minimum_size(config):
    bars = make_bars(
        [
            (100.0, 100.5, 99.5, 100.0),   # C1, high 100.5
            (100.0, 102.0, 100.0, 101.8),  # C2 displacement
            (101.8, 102.5, 101.0, 102.2),  # C3, low 101.0
        ],
        atr=1.0,
    )
    engine, _ = run_engine(bars, config)
    fvgs = [z for z in engine.all_zones if z.type is ZoneType.FVG]
    assert len(fvgs) == 1
    zone = fvgs[0]
    assert zone.direction == "BUY"
    assert zone.low == pytest.approx(100.5)   # High(C1)
    assert zone.high == pytest.approx(101.0)  # Low(C3)


def test_bearish_fvg_boundaries(config):
    bars = make_bars(
        [
            (102.0, 102.5, 101.5, 102.0),  # C1, low 101.5
            (102.0, 102.0, 100.0, 100.2),
            (100.2, 101.0, 99.5, 99.8),    # C3, high 101.0
        ],
        atr=1.0,
    )
    engine, _ = run_engine(bars, config)
    zone = [z for z in engine.all_zones if z.type is ZoneType.FVG][0]
    assert zone.direction == "SELL"
    assert zone.low == pytest.approx(101.0)   # High(C3)
    assert zone.high == pytest.approx(101.5)  # Low(C1)


def test_fvg_below_minimum_size_is_rejected(config):
    """Gap of 0.05 with ATR 1.0 is under the 0.10 x ATR14 floor."""
    bars = make_bars(
        [
            (100.0, 100.50, 99.5, 100.0),
            (100.0, 101.00, 100.0, 100.8),
            (100.8, 101.20, 100.55, 101.0),  # Low(C3) 100.55 - High(C1) 100.50 = 0.05
        ],
        atr=1.0,
    )
    engine, _ = run_engine(bars, config)
    assert not [z for z in engine.all_zones if z.type is ZoneType.FVG]


def test_fvg_partial_fill_stays_valid_but_close_through_far_boundary_invalidates(config):
    base = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 102.0, 100.0, 101.8),
        (101.8, 102.5, 101.0, 102.2),  # bullish FVG [100.5, 101.0]
    ]
    partial = base + [(102.2, 102.3, 100.7, 100.8)]  # fills part of the gap, closes inside it
    engine, _ = run_engine(make_bars(partial, atr=1.0), config)
    zone = [z for z in engine.all_zones if z.type is ZoneType.FVG][0]
    assert not zone.retired
    assert zone.touched

    broken = base + [(102.2, 102.3, 100.0, 100.2)]  # closes below High(C1) = far boundary
    engine, _ = run_engine(make_bars(broken, atr=1.0), config)
    zone = [z for z in engine.all_zones if z.type is ZoneType.FVG][0]
    assert zone.retired and zone.retired_reason == "invalidated"


# ----------------------------------------------------------------- OB ----
OB_PREFIX = [
    (100.0, 100.5, 99.5, 100.2),
    (100.2, 101.0, 100.0, 100.8),
    (100.8, 101.5, 100.5, 101.2),
    (101.2, 103.0, 101.0, 102.5),   # index 3: swing high at 103.0
    (102.5, 102.6, 101.5, 101.8),
    (101.8, 102.0, 101.0, 101.2),   # index 5: pivot at 3 becomes confirmed
    (101.2, 101.5, 100.5, 100.8),   # index 6: last bearish candle -> OB candidate
]


def test_bullish_order_block_requires_displacement_and_swing_break(config):
    bars = make_bars(OB_PREFIX + [(100.8, 104.0, 100.7, 103.5)], atr=1.0)
    engine, _ = run_engine(bars, config)
    obs = [z for z in engine.all_zones if z.type is ZoneType.OB]
    assert len(obs) == 1
    zone = obs[0]
    assert zone.direction == "BUY"
    # OB boundaries use the entire qualifying candle, wicks included
    assert (zone.low, zone.high) == pytest.approx((100.5, 101.5))
    assert zone.created_bar == 7  # created when the displacement confirms it


def test_order_block_rejected_without_swing_break(config):
    """Displacement of 1.2 ATR is enough, but the close stays under the swing high."""
    bars = make_bars(OB_PREFIX + [(100.8, 102.2, 100.7, 102.0)], atr=1.0)
    engine, _ = run_engine(bars, config)
    assert not [z for z in engine.all_zones if z.type is ZoneType.OB]


def test_order_block_rejected_with_insufficient_displacement(config):
    """Close breaks the swing high but the move is only 0.9 x ATR14."""
    bars = make_bars(OB_PREFIX + [(100.8, 103.5, 100.7, 103.2)], atr=2.7)
    engine, _ = run_engine(bars, config)
    assert not [z for z in engine.all_zones if z.type is ZoneType.OB]


def test_order_block_displacement_window_expires_after_three_bars(config):
    tail = [
        (100.8, 101.0, 100.6, 100.9),
        (100.9, 101.1, 100.7, 101.0),
        (101.0, 101.2, 100.8, 101.1),
        (101.1, 104.0, 101.0, 103.9),  # 4th bar after the candidate -- too late
    ]
    bars = make_bars(OB_PREFIX + tail, atr=1.0)
    engine, _ = run_engine(bars, config)
    ob_from_candidate = [z for z in engine.all_zones if z.type is ZoneType.OB and z.low == pytest.approx(100.5)]
    assert not ob_from_candidate


def test_order_block_wick_alone_does_not_invalidate(config):
    tail = [
        (100.8, 104.0, 100.7, 103.5),   # OB [100.5, 101.5] confirmed here
        (103.5, 103.6, 100.2, 101.0),   # deep wick below OB low, closes back inside the OB
    ]
    engine, _ = run_engine(make_bars(OB_PREFIX + tail, atr=1.0), config)
    ob = [z for z in engine.all_zones if z.type is ZoneType.OB][0]
    assert not ob.retired

    tail_close_through = [
        (100.8, 104.0, 100.7, 103.5),
        (103.5, 103.6, 100.2, 100.3),   # closes below OB low
    ]
    engine, _ = run_engine(make_bars(OB_PREFIX + tail_close_through, atr=1.0), config)
    ob = [z for z in engine.all_zones if z.type is ZoneType.OB][0]
    assert ob.retired and ob.retired_reason == "invalidated"


# ----------------------------------------------------------------- S/R ----
def test_support_zone_from_confirmed_two_left_two_right_pivot(config):
    bars = make_bars(
        [
            (100.0, 100.5, 99.8, 100.1),
            (100.1, 100.4, 99.6, 99.9),
            (99.9, 100.0, 99.0, 99.4),   # index 2: pivot low at 99.0
            (99.4, 100.2, 99.3, 100.0),
            (100.0, 100.8, 99.7, 100.6),  # index 4: pivot confirmed here
        ],
        atr=1.0,
    )
    engine, _ = run_engine(bars, config)
    sr = [z for z in engine.all_zones if z.type is ZoneType.SR]
    assert len(sr) == 1
    zone = sr[0]
    assert zone.direction == "BUY"
    assert (zone.low, zone.high) == pytest.approx((99.0 - 0.10, 99.0 + 0.10))
    assert zone.created_bar == 4  # not knowable before the 2 right-hand bars close


def test_pivot_not_confirmed_without_two_right_bars(config):
    bars = make_bars(
        [
            (100.0, 100.5, 99.8, 100.1),
            (100.1, 100.4, 99.6, 99.9),
            (99.9, 100.0, 99.0, 99.4),
            (99.4, 100.2, 99.3, 100.0),  # only one right-hand bar exists
        ],
        atr=1.0,
    )
    engine, _ = run_engine(bars, config)
    assert not [z for z in engine.all_zones if z.type is ZoneType.SR]


# ------------------------------------------------------- lifecycle ----
def _bullish_fvg_bars(tail):
    return make_bars(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 102.0, 100.0, 101.8),
            (101.8, 102.5, 101.0, 102.2),  # bullish FVG [100.5, 101.0]
        ]
        + tail,
        atr=1.0,
    )


def test_valid_buy_reaction_within_two_bars_emits_event(config):
    tail = [
        (102.2, 102.3, 100.8, 100.9),  # touch: enters the zone, closes inside
        (100.9, 101.4, 100.8, 101.3),  # closes above near/upper boundary 101.0
    ]
    engine, events = run_engine(_bullish_fvg_bars(tail), config)
    assert len(events) == 1
    event = events[0]
    assert event.direction == "BUY"
    assert event.touch_bar == 3 and event.reaction_bar == 4
    assert event.zone.retired and event.zone.retired_reason == "reacted"


def test_reaction_later_than_two_bars_after_touch_is_not_accepted(config):
    tail = [
        (102.2, 102.3, 100.8, 100.9),  # touch at bar 3
        (100.9, 100.95, 100.7, 100.9),
        (100.9, 100.95, 100.7, 100.9),
        (100.9, 101.5, 100.8, 101.4),  # bar 6: three bars after the touch
    ]
    engine, events = run_engine(_bullish_fvg_bars(tail), config)
    assert events == []


def test_first_reaction_only_retires_the_zone(config):
    tail = [
        (102.2, 102.3, 100.8, 100.9),
        (100.9, 101.4, 100.8, 101.3),  # first reaction -> retired
        (101.3, 101.4, 100.8, 100.9),  # touches again
        (100.9, 101.6, 100.8, 101.5),  # would have reacted again
    ]
    engine, events = run_engine(_bullish_fvg_bars(tail), config)
    assert len(events) == 1


def test_untouched_zone_expires_after_max_age_bars(config):
    filler = [(102.2, 102.4, 102.0, 102.2)] * (config.zone_max_age_bars + 1)
    engine, _ = run_engine(_bullish_fvg_bars(filler), config)
    zone = [z for z in engine.all_zones if z.type is ZoneType.FVG][0]
    assert zone.retired and zone.retired_reason == "expired"


def test_zone_age_shorter_than_limit_stays_eligible(config):
    filler = [(102.2, 102.4, 102.0, 102.2)] * (config.zone_max_age_bars - 5)
    engine, _ = run_engine(_bullish_fvg_bars(filler), config)
    zone = [z for z in engine.all_zones if z.type is ZoneType.FVG][0]
    assert not zone.retired


def test_confluence_score_counts_distinct_overlapping_zone_types(config):
    """An S/R support zone overlapping the FVG lifts the score from 1 to 2."""
    bars = make_bars(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 102.0, 100.0, 101.8),
            (101.8, 102.5, 101.0, 102.2),    # bullish FVG [100.5, 101.0]
            (102.2, 102.6, 102.0, 102.4),
            (102.4, 102.8, 102.2, 102.5),
            (102.5, 102.9, 102.3, 102.6),
            (102.6, 102.7, 101.05, 102.4),   # index 6: pivot low 101.05, clear of the FVG
            (102.4, 102.6, 102.0, 102.5),
            (102.5, 102.7, 102.1, 102.6),    # index 8: pivot confirmed -> SR zone [100.95, 101.15]
            (102.6, 102.7, 100.8, 101.05),   # touch + reaction close above the FVG's 101.0
        ],
        atr=1.0,
    )
    engine, events = run_engine(bars, config)
    assert events, "expected a reaction event"
    fvg_events = [e for e in events if e.zone.type is ZoneType.FVG]
    assert fvg_events, "expected the FVG to be the reacting zone"
    assert fvg_events[0].confluence_score >= 2
    assert "SR" in fvg_events[0].confluence_types
