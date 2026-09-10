"""Section 5: Risk Management and Position Sizing.

Implements the structural stop-loss, the fixed initial-balance risk
model, broker-constrained dynamic lot sizing, and the exact 3R take
profit target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class StopLossResult:
    stop_price: float
    stop_distance: float
    buffer: float
    base_reference: float


@dataclass
class LotSizeResult:
    accepted: bool
    reason: str
    raw_lots: float
    normalized_lots: float
    monetary_loss_per_lot: float
    theoretical_risk: float
    normalized_risk: float


@dataclass
class TradePlan:
    direction: str
    entry: float
    stop_loss: StopLossResult
    take_profit: float
    lot_size: LotSizeResult


def compute_stop_loss(
    direction: str,
    zone_far_boundary: float,
    reaction_extreme: float,
    atr: float,
    spread: float,
    sl_buffer_atr: float,
    sl_buffer_spread_mult: float,
) -> StopLossResult:
    """Section 5.4: structural stop beyond the farther of zone/reaction extreme."""
    buffer = max(sl_buffer_atr * atr, sl_buffer_spread_mult * spread)
    if direction == "BUY":
        base = min(zone_far_boundary, reaction_extreme)
        stop_price = base - buffer
    elif direction == "SELL":
        base = max(zone_far_boundary, reaction_extreme)
        stop_price = base + buffer
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return StopLossResult(stop_price=stop_price, stop_distance=abs(base - stop_price) + 0.0, buffer=buffer, base_reference=base)


def stop_distance_from_entry(direction: str, entry: float, stop_loss_price: float) -> float:
    if direction == "BUY":
        return entry - stop_loss_price
    return stop_loss_price - entry


def compute_take_profit(direction: str, entry: float, stop_distance: float, reward_risk: float) -> float:
    """Section 5.4: baseline TP is exactly reward_risk * stop_distance (3R)."""
    if direction == "BUY":
        return entry + reward_risk * stop_distance
    return entry - reward_risk * stop_distance


def compute_lot_size(
    stop_distance: float,
    risk_budget: float,
    broker,
) -> LotSizeResult:
    """Section 5.3: dynamic, broker-normalized lot sizing from the structural stop.

    ``broker`` is a :class:`~xauusd_robot.config.BrokerSpec`. Money at risk
    for 1.0 lot is derived from broker tick size/tick value rather than an
    assumed universal XAUUSD point value.
    """
    if stop_distance <= 0:
        return LotSizeResult(False, "invalid_stop_distance", 0.0, 0.0, 0.0, 0.0, 0.0)

    ticks = stop_distance / broker.tick_size
    monetary_loss_per_lot = ticks * broker.tick_value

    risk_at_min_lot = broker.volume_min * monetary_loss_per_lot
    if risk_at_min_lot > risk_budget:
        return LotSizeResult(
            False, "min_lot_exceeds_risk_cap", 0.0, 0.0, monetary_loss_per_lot, risk_budget, risk_at_min_lot
        )

    raw_lots = risk_budget / monetary_loss_per_lot
    steps = math.floor(raw_lots / broker.volume_step + 1e-9)
    normalized_lots = steps * broker.volume_step
    normalized_lots = max(normalized_lots, broker.volume_min)
    normalized_lots = min(normalized_lots, broker.volume_max)
    normalized_lots = round(normalized_lots, 8)

    normalized_risk = normalized_lots * monetary_loss_per_lot
    if normalized_risk > risk_budget + 1e-9:
        return LotSizeResult(
            False, "normalized_risk_exceeds_cap", raw_lots, normalized_lots,
            monetary_loss_per_lot, risk_budget, normalized_risk,
        )

    return LotSizeResult(
        True, "ok", raw_lots, normalized_lots, monetary_loss_per_lot, risk_budget, normalized_risk
    )


def build_trade_plan(
    direction: str,
    entry: float,
    zone_far_boundary: float,
    reaction_extreme: float,
    atr: float,
    spread: float,
    initial_balance: float,
    config,
) -> TradePlan:
    sl = compute_stop_loss(
        direction, zone_far_boundary, reaction_extreme, atr, spread,
        config.sl_buffer_atr, config.sl_buffer_spread_mult,
    )
    distance = stop_distance_from_entry(direction, entry, sl.stop_price)
    sl.stop_distance = distance
    tp = compute_take_profit(direction, entry, distance, config.reward_risk)
    risk_budget = initial_balance * config.risk_fraction()
    lot = compute_lot_size(distance, risk_budget, config.broker)
    return TradePlan(direction=direction, entry=entry, stop_loss=sl, take_profit=tp, lot_size=lot)
