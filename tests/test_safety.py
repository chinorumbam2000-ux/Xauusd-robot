"""Section 5.5 acceptance tests: circuit breakers, locks, cooldown, restart."""
from __future__ import annotations

from datetime import date

import pytest

from xauusd_robot.config import StrategyConfig
from xauusd_robot.safety import SafetyEngine, SafetyState


def engine(balance=1000.0, config=None):
    config = config or StrategyConfig()
    state = SafetyState.new(balance)
    state.broker_day = date(2024, 1, 1).isoformat()
    return SafetyEngine(config, state)


ARGS = dict(spread=0.05, sl_distance=1.0, atr=1.0)


def test_baseline_setup_is_allowed():
    assert engine().can_open_new_trade(100, **ARGS) == (True, "ok")


def test_only_one_position_at_a_time():
    eng = engine()
    eng.register_trade_open()
    assert eng.can_open_new_trade(100, **ARGS) == (False, "position_open")


def test_fourth_trade_of_the_day_is_blocked():
    eng = engine()
    for i in range(3):
        eng.register_trade_open()
        eng.register_trade_close(realized_r=3.0, closed_balance=1000.0 + i, current_bar_index=i)
    assert eng.state.trades_today == 3
    assert eng.can_open_new_trade(100, **ARGS)[1] == "max_trades_per_day"


def test_second_losing_trade_blocks_new_entries():
    eng = engine()
    for i in range(2):
        eng.register_trade_open()
        eng.register_trade_close(realized_r=-1.0, closed_balance=990.0 - i, current_bar_index=i)
    assert eng.state.losses_today == 2
    assert eng.can_open_new_trade(100, **ARGS)[1] == "max_losses_per_day"


def test_daily_minus_two_r_blocks_new_entries():
    """Reached via -0.9R + -1.1R, i.e. before the losing-trade count trips."""
    config = StrategyConfig(max_losses_per_day=99)
    eng = engine(config=config)
    eng.register_trade_open()
    eng.register_trade_close(realized_r=-0.9, closed_balance=991.0, current_bar_index=0)
    eng.register_trade_open()
    eng.register_trade_close(realized_r=-1.1, closed_balance=980.0, current_bar_index=1)
    assert eng.state.daily_realized_r == pytest.approx(-2.0)
    assert eng.can_open_new_trade(100, **ARGS)[1] == "daily_loss_limit"


def test_daily_counters_reset_on_a_new_broker_day():
    eng = engine()
    for i in range(3):
        eng.register_trade_open()
        eng.register_trade_close(realized_r=-1.0, closed_balance=950.0, current_bar_index=i)
    eng.roll_broker_day(date(2024, 1, 2))
    assert (eng.state.trades_today, eng.state.losses_today, eng.state.daily_realized_r) == (0, 0, 0.0)
    assert eng.can_open_new_trade(100, **ARGS)[0]


def test_cooldown_blocks_entries_for_three_bars_after_a_close():
    eng = engine()
    eng.register_trade_open()
    eng.register_trade_close(realized_r=3.0, closed_balance=1150.0, current_bar_index=50)
    assert eng.state.cooldown_until_bar == 53
    assert eng.can_open_new_trade(52, **ARGS)[1] == "cooldown"
    assert eng.can_open_new_trade(53, **ARGS)[0]


def test_fifteen_percent_peak_equity_drawdown_locks_new_entries():
    eng = engine()
    eng.update_equity(2000.0)          # new peak
    eng.update_equity(1701.0)          # -14.95%, still trading
    assert not eng.state.drawdown_locked
    eng.update_equity(1700.0)          # -15.0%
    assert eng.state.drawdown_locked
    assert eng.can_open_new_trade(100, **ARGS)[1] == "drawdown_lock"


def test_drawdown_lock_requires_a_manual_reset():
    eng = engine()
    eng.update_equity(2000.0)
    eng.update_equity(1500.0)
    assert eng.state.drawdown_locked
    eng.update_equity(2100.0)  # recovery alone must not unlock it
    assert eng.state.drawdown_locked
    eng.manual_reset_drawdown_lock()
    assert not eng.state.drawdown_locked


def test_ten_x_target_uses_closed_balance_only():
    eng = engine()
    eng.update_equity(12000.0)  # floating equity above target must not trigger it
    assert not eng.state.target_reached
    eng.register_trade_open()
    eng.register_trade_close(realized_r=3.0, closed_balance=10000.0, current_bar_index=10)
    assert eng.state.target_reached
    assert eng.can_open_new_trade(100, **ARGS)[1] == "target_reached"


def test_spread_filter_requires_both_tests_to_pass():
    eng = engine()
    # spread <= 10% of SL distance, but > 0.15 x ATR
    assert eng.can_open_new_trade(100, spread=0.09, sl_distance=1.0, atr=0.5)[1] == "spread_vs_atr"
    # spread within the ATR test but too large against the stop distance
    assert eng.can_open_new_trade(100, spread=0.09, sl_distance=0.5, atr=1.0)[1] == "spread_vs_sl"
    assert eng.can_open_new_trade(100, spread=0.09, sl_distance=1.0, atr=1.0)[0]


def test_state_survives_a_restart_round_trip():
    eng = engine()
    eng.update_equity(1800.0)
    eng.register_trade_open()
    eng.register_trade_close(realized_r=-1.0, closed_balance=1750.0, current_bar_index=42)

    restored = SafetyState.from_json(eng.state.to_json())
    assert restored.initial_balance == 1000.0
    assert restored.running_equity_peak == pytest.approx(1800.0)
    assert restored.cooldown_until_bar == 45
    assert restored.losses_today == 1
    assert restored.closed_balance == pytest.approx(1750.0)
