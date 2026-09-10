"""Configuration parameters, Section 13 of the Blueprint v1.1.

All defaults below mirror the "Initial / Test Value" column of the
blueprint's configuration table exactly. Nothing here should be changed
casually -- the blueprint freezes these as the v1.1 baseline. Use
``dataclasses.replace(config, **overrides)`` to build variant configs for
robustness / sensitivity testing (Section 10) rather than mutating shared
instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BrokerSpec:
    """Broker-reported XAUUSD symbol properties (Section 5.3 / 6.1 MarketData).

    In live MQL5 these must be read dynamically via SymbolInfoDouble /
    SymbolInfoInteger. In the Python research layer we accept them as
    explicit, auditable inputs so backtests are reproducible.
    """

    symbol: str = "XAUUSD"
    contract_size: float = 100.0  # oz per 1.0 lot, standard XAUUSD contract
    tick_size: float = 0.01  # minimum price increment
    tick_value: float = 1.0  # account-currency value of one tick_size move, per 1.0 lot
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    leverage: float = 100.0  # used only for the "margin unavailable" rejection test
    default_spread_points: float = 20.0  # 0.20 price units, used when no spread column present


@dataclass(frozen=True)
class StrategyConfig:
    """Section 13 configuration parameters, deterministic v1.1 baseline."""

    # --- Instrument / timeframe ---
    symbol: str = "XAUUSD"
    execution_tf: str = "M5"
    regime_timeframes: Tuple[str, ...] = ("D1", "H4", "H1", "M30", "M15", "M5")

    # --- Indicators ---
    ema_period: int = 200
    wpr_period: int = 49
    wpr_oversold: float = -80.0
    wpr_overbought: float = -20.0
    wpr_max_lead_bars: int = 2
    atr_period: int = 14

    # --- Order Block ---
    ob_displacement_atr: float = 1.0
    ob_displacement_bars: int = 3
    ob_require_swing_break: bool = True

    # --- Zone aging / lifecycle ---
    zone_max_age_bars: int = 96
    first_reaction_only: bool = True

    # --- Support / Resistance ---
    sr_lookback: int = 20
    sr_pivot_left: int = 2
    sr_pivot_right: int = 2
    sr_zone_atr: float = 0.10

    # --- Fair Value Gap ---
    fvg_min_atr: float = 0.10

    # --- Reaction / setup lifecycle ---
    reaction_max_bars: int = 2
    setup_expiry_bars: int = 5

    # --- Push candles ---
    push_body_ratio: float = 0.60
    push2_breaks_push1: bool = True

    # --- Stop-loss buffer ---
    sl_buffer_atr: float = 0.10
    sl_buffer_spread_mult: float = 2.0

    # --- Risk / reward ---
    risk_percent_initial_balance: float = 5.0  # % ; test matrix: 0.5/1/2/3/5
    reward_risk: float = 3.0

    # --- Safety / circuit breakers ---
    max_open_positions: int = 1
    max_trades_per_day: int = 3
    max_losses_per_day: int = 2
    daily_loss_limit_r: float = -2.0
    max_peak_equity_drawdown: float = 0.15
    max_spread_vs_sl: float = 0.10
    max_spread_atr: float = 0.15
    cooldown_bars: int = 3
    target_multiplier: float = 10.0

    # --- Experimental filters (OFF in baseline v1.1) ---
    use_session_filter: bool = False
    use_news_filter: bool = False

    broker: BrokerSpec = field(default_factory=BrokerSpec)

    def risk_fraction(self) -> float:
        return self.risk_percent_initial_balance / 100.0


#: Research matrix from Section 10 / 17.2 -- compare, never cherry-pick.
RISK_TEST_MATRIX: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)

DEFAULT_CONFIG = StrategyConfig()
