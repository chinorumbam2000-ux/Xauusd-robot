"""Section 5 acceptance tests: structural stop, risk cap, minimum lot, 1:3 TP."""
from __future__ import annotations

from dataclasses import replace

import pytest

from xauusd_robot.config import BrokerSpec, StrategyConfig
from xauusd_robot.risk import (
    build_trade_plan,
    compute_lot_size,
    compute_stop_loss,
    compute_take_profit,
)


def test_buy_stop_sits_below_the_lower_of_zone_edge_and_reaction_low():
    result = compute_stop_loss(
        "BUY", zone_far_boundary=100.0, reaction_extreme=99.8,
        atr=1.0, spread=0.05, sl_buffer_atr=0.10, sl_buffer_spread_mult=2.0,
    )
    assert result.base_reference == pytest.approx(99.8)   # the farther reference
    assert result.buffer == pytest.approx(0.10)           # max(0.10 x 1.0, 2 x 0.05)
    assert result.stop_price == pytest.approx(99.7)


def test_sell_stop_sits_above_the_higher_of_zone_edge_and_reaction_high():
    result = compute_stop_loss(
        "SELL", zone_far_boundary=101.0, reaction_extreme=101.4,
        atr=1.0, spread=0.05, sl_buffer_atr=0.10, sl_buffer_spread_mult=2.0,
    )
    assert result.base_reference == pytest.approx(101.4)
    assert result.stop_price == pytest.approx(101.5)


def test_buffer_uses_spread_when_spread_dominates_atr():
    """Buffer = max(0.10 x ATR14, 2 x spread) -- the spread branch."""
    result = compute_stop_loss(
        "BUY", zone_far_boundary=100.0, reaction_extreme=100.0,
        atr=1.0, spread=0.40, sl_buffer_atr=0.10, sl_buffer_spread_mult=2.0,
    )
    assert result.buffer == pytest.approx(0.80)
    assert result.stop_price == pytest.approx(99.2)


def test_take_profit_is_exactly_three_times_the_stop_distance():
    assert compute_take_profit("BUY", entry=100.0, stop_distance=0.5, reward_risk=3.0) == pytest.approx(101.5)
    assert compute_take_profit("SELL", entry=100.0, stop_distance=0.5, reward_risk=3.0) == pytest.approx(98.5)


# --------------------------------------------------------------- sizing ----
BROKER = BrokerSpec()  # tick_size 0.01, tick_value 1.0 -> $100 per 1.00 price move per lot


def test_lot_size_matches_the_risk_budget_exactly_when_divisible():
    result = compute_lot_size(stop_distance=1.0, risk_budget=50.0, broker=BROKER)
    assert result.accepted
    assert result.monetary_loss_per_lot == pytest.approx(100.0)
    assert result.normalized_lots == pytest.approx(0.5)
    assert result.normalized_risk == pytest.approx(50.0)


def test_lot_size_rounds_down_so_the_risk_cap_is_never_exceeded():
    result = compute_lot_size(stop_distance=1.03, risk_budget=50.0, broker=BROKER)
    assert result.accepted
    assert result.raw_lots == pytest.approx(50.0 / 103.0)
    assert result.normalized_lots == pytest.approx(0.48)  # 0.4854... rounded DOWN
    assert result.normalized_risk <= 50.0


def test_trade_is_rejected_when_minimum_lot_would_exceed_the_risk_cap():
    result = compute_lot_size(stop_distance=10.0, risk_budget=1.0, broker=BROKER)
    assert not result.accepted
    assert result.reason == "min_lot_exceeds_risk_cap"


def test_zero_or_negative_stop_distance_is_rejected():
    assert not compute_lot_size(stop_distance=0.0, risk_budget=50.0, broker=BROKER).accepted


def test_volume_is_capped_at_broker_maximum():
    broker = replace(BROKER, volume_max=0.10)
    result = compute_lot_size(stop_distance=1.0, risk_budget=5000.0, broker=broker)
    assert result.normalized_lots == pytest.approx(0.10)


# ----------------------------------------------------- fixed-balance risk ----
def test_risk_is_fixed_to_initial_balance_not_current_balance():
    """Section 5.2: $50 on a $1,000 start stays $50 as the account grows."""
    config = replace(StrategyConfig(), risk_percent_initial_balance=5.0)
    plan_start = build_trade_plan(
        "BUY", entry=100.5, zone_far_boundary=100.0, reaction_extreme=100.0,
        atr=1.0, spread=0.05, initial_balance=1000.0, config=config,
    )
    assert plan_start.lot_size.theoretical_risk == pytest.approx(50.0)

    # A grown account still risks the same dollars off the INITIAL balance.
    plan_later = build_trade_plan(
        "BUY", entry=100.5, zone_far_boundary=100.0, reaction_extreme=100.0,
        atr=1.0, spread=0.05, initial_balance=1000.0, config=config,
    )
    assert plan_later.lot_size.theoretical_risk == pytest.approx(50.0)


def test_effective_risk_percentage_decays_as_the_balance_grows():
    config = StrategyConfig()
    risk_dollars = 1000.0 * config.risk_fraction()
    assert risk_dollars / 1000.0 == pytest.approx(0.05)
    assert risk_dollars / 2000.0 == pytest.approx(0.025)
    assert risk_dollars / 10000.0 == pytest.approx(0.005)


def test_trade_plan_target_is_three_r_from_the_actual_entry():
    config = StrategyConfig()
    plan = build_trade_plan(
        "BUY", entry=100.5, zone_far_boundary=100.0, reaction_extreme=99.9,
        atr=1.0, spread=0.05, initial_balance=1000.0, config=config,
    )
    distance = plan.entry - plan.stop_loss.stop_price
    assert plan.take_profit - plan.entry == pytest.approx(3.0 * distance)
