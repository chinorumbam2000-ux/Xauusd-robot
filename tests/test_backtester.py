"""End-to-end harness tests: Section 15 invariants must hold over a whole run.

These use a shortened EMA period so a modest fixture still warms up (a true
EMA200 on D1 needs ~57,600 M5 bars before the first tradeable bar). The
rules under test -- risk cap, daily limits, one position, exact 3R -- are
unaffected by that choice.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_sample_data import generate  # noqa: E402

from xauusd_robot.backtester import Backtester  # noqa: E402
from xauusd_robot.config import StrategyConfig  # noqa: E402
from xauusd_robot.safety import SafetyState  # noqa: E402


@pytest.fixture(scope="module")
def run():
    frame = generate(bars=60000, seed=7)
    m5 = frame.set_index("time")
    m5["spread"] = 10
    config = replace(StrategyConfig(), ema_period=50, risk_percent_initial_balance=1.0)
    return Backtester(m5, config, initial_balance=1000.0).run()


def test_backtest_produces_trades_and_a_complete_audit_trail(run):
    assert run.metrics["trades"] > 0
    assert not run.events.empty
    assert {"setup_armed", "order_placed", "trade_closed"} <= set(run.events["event"])
    assert len(run.equity) == 60000  # one equity snapshot per M5 bar


def test_every_trade_respects_the_initial_balance_risk_cap(run):
    """Section 5.2/5.3: risk is 1% of the INITIAL balance, never more."""
    budget = 1000.0 * 0.01
    assert (run.trades["risk_money"] <= budget + 1e-9).all()


def test_losses_are_capped_at_one_r_and_wins_pay_three_r(run):
    stops = run.trades[run.trades["exit_reason"] == "stop_loss"]
    targets = run.trades[run.trades["exit_reason"] == "take_profit"]
    assert (stops["r_multiple"] <= -0.999).all()
    assert (stops["r_multiple"] >= -3.0).all()  # gap fills are worse, but bounded
    if not targets.empty:
        assert targets["r_multiple"].min() >= 2.999


def test_take_profit_distance_is_exactly_three_times_the_stop_distance(run):
    for _, t in run.trades.iterrows():
        reward = abs(t["target_price"] - t["entry_price"])
        assert reward == pytest.approx(3.0 * t["stop_distance"], rel=1e-6)


def test_never_more_than_one_position_at_a_time(run):
    trades = run.trades.sort_values("entry_time")
    assert (trades["entry_time"].shift(-1).dropna() >= trades["exit_time"][:-1].values).all()


def test_daily_trade_and_loss_limits_are_never_breached(run):
    per_day = run.trades.groupby(run.trades["exit_time"].dt.date)
    assert per_day.size().max() <= 3
    losses = run.trades[run.trades["r_multiple"] < 0]
    if not losses.empty:
        assert losses.groupby(losses["exit_time"].dt.date).size().max() <= 2


def test_cooldown_of_three_bars_is_honoured_between_trades(run):
    trades = run.trades.sort_values("entry_time")
    gaps = trades["entry_time"].shift(-1) - trades["exit_time"]
    assert (gaps.dropna() >= pd.Timedelta(minutes=15)).all()


def test_equity_curve_reconciles_with_the_trade_ledger(run):
    expected = 1000.0 + run.trades["pnl_money"].sum()
    assert run.final_balance == pytest.approx(expected)
    assert run.trades["balance_after"].iloc[-1] == pytest.approx(expected)


def test_funnel_explains_where_setups_died(run):
    funnel = run.metrics["funnel"]
    assert funnel["bars"] == 60000
    assert funnel["setups_armed"] >= funnel["setups_entry_evaluated"]
    assert funnel["orders_placed"] == run.metrics["trades"]
    assert funnel["zone_reactions"] >= funnel["zone_reactions_regime_matched"]


def test_no_trade_is_taken_on_a_mixed_regime_bar(run):
    placed = run.events[run.events["event"] == "order_placed"]
    assert (placed["regime"] != "MIXED").all()
    assert placed["direction"].equals(placed["regime"])


def test_run_can_resume_from_a_persisted_safety_state():
    """Section 12 / "Restart": locks and references survive a reload."""
    frame = generate(bars=60000, seed=7)
    m5 = frame.set_index("time")
    config = replace(StrategyConfig(), ema_period=50)

    state = SafetyState.new(1000.0)
    state.drawdown_locked = True  # simulate a lock carried across a restart
    restored = SafetyState.from_json(state.to_json())
    result = Backtester(m5, config, 1000.0, safety_state=restored).run()

    assert result.metrics["trades"] == 0  # locked out until a manual reset
    assert result.safety_state.drawdown_locked
