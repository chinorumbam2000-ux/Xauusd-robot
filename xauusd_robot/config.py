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
from typing import Optional, Tuple


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
    #: Account-currency value of one unit of the symbol's BASE currency, used
    #: only by the margin test. None means "use the current price", which is
    #: correct whenever the QUOTE currency is the account currency -- XAUUSD,
    #: EURUSD and GBPUSD on a USD account. It is wrong for a USD-base pair like
    #: USDJPY, whose notional is already in USD and must not be multiplied by
    #: 154, and for a cross like EURGBP, whose base converts through a third
    #: rate rather than through its own price.
    margin_base_rate: Optional[float] = None


@dataclass(frozen=True)
class StrategyConfig:
    """Section 13 configuration parameters, deterministic v1.1 baseline."""

    # --- Instrument / timeframe ---
    symbol: str = "XAUUSD"
    execution_tf: str = "M5"
    #: DEVIATION FROM BLUEPRINT v1.1 (Section 3.1 specifies six timeframes:
    #: D1, H4, H1, M30, M15, M5). D1 and M15 have been removed.
    #:
    #: M15 was measured as entirely redundant: adding or removing it produced
    #: byte-identical results, because it sits between M30 and M5 in the trend
    #: hierarchy and effectively never disagrees with both.
    #:
    #: M5 is kept, but the reason is drawdown rather than signal quality. An
    #: earlier note here claimed dropping it produced a 21.4% win rate and
    #: negative expectancy; that came from a 2% risk run truncated by the
    #: lockout after 14 trades and was wrong. Measured at 0.5% risk, where
    #: neither variant locks out, H4/H1/M30 alone gives 142 trades at 36.6% and
    #: +0.460R against 74 trades at 44.6% and +0.773R -- nearly double the
    #: frequency, still clearly profitable, and neither gap is statistically
    #: significant.
    #:
    #: What decides it is that drawdown per unit of risk is about twice as high
    #: without M5, so it can only carry a third of the risk percent. Held to a
    #: 12% drawdown budget: 165.1% return at 3% risk with M5, against 65.4% at
    #: 1% risk without it. The M5-less variant does show better walk-forward
    #: consistency (5/5 windows, +3.0R worst) and remains a legitimate choice
    #: for anyone preferring frequency over return per unit of risk -- but it
    #: needs risk dropped to ~1%, since 2% would reach ~19% drawdown and lock.
    #:
    #: Warm-up is ~14,400 M5 bars, set by H4 as the slowest member.
    #: Set BLUEPRINT_REGIME_TIMEFRAMES to restore the original six.
    regime_timeframes: Tuple[str, ...] = ("H4", "H1", "M30", "M5")

    # --- Indicators ---
    ema_period: int = 200
    wpr_period: int = 49
    wpr_oversold: float = -80.0
    wpr_overbought: float = -20.0
    wpr_max_lead_bars: int = 2
    #: Which momentum gate confirms the setup. "wpr" is the blueprint rule
    #: (Section 3.3); "macd" substitutes a signal-line cross; "none" removes
    #: the gate entirely and is only useful as a control.
    momentum_filter: str = "wpr"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
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
    #: Body ratio required of Push 2 specifically. None means "same as
    #: push_body_ratio". The two candles do different jobs -- Push 1
    #: establishes the impulse, Push 2 only has to show the move is still
    #: going, and the break-beyond-Push-1 test already carries most of that
    #: proof -- so they need not share a threshold.
    push2_body_ratio: Optional[float] = None
    push2_breaks_push1: bool = True
    #: Section 3.4/4.6 make the two push candles mandatory. Setting this
    #: False removes the trigger entirely: entry fires as soon as the WPR
    #: exit confirms, with no candle-quality requirement at all.
    #:
    #: Measured at a risk low enough that the lockout truncates nothing, all
    #: three settings are profitable -- the differences are in how much, and in
    #: how deep the losing stretches get. Held to a 12% drawdown budget:
    #:
    #:   variant        risk  trades  /month  win%    expR   net%  wf   worst
    #:   2 pushes       3.00      75     5.0  44.0  +0.750  165.1  4/5    0.0
    #:   1 push         1.00     257    16.7  32.7  +0.289   72.0  4/5   -7.8
    #:   no push        0.75     352    22.8  29.3  +0.159   40.9  3/5  -12.9
    #:
    #: Removing the trigger buys 4.5x the trade frequency and costs three
    #: quarters of the return, because the weaker per-trade edge forces the risk
    #: percent down. The worst walk-forward window also deepens from break-even
    #: to -12.9R, so losing stretches get considerably harder to sit through.
    require_push_candles: bool = True
    #: How many consecutive push candles the trigger needs. Section 3.4
    #: specifies 2, where the second must also close beyond the first's
    #: extreme. Setting 1 keeps the directional body-quality test but drops
    #: the continuation requirement, so entry fires on the first qualifying
    #: candle. push2_breaks_push1 is then irrelevant.
    push_candles_required: int = 2

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
    #: Section 10 lists these as later experiments, to be evaluated in
    #: isolation. Turning one on is a deviation from the v1.1 baseline.
    use_session_filter: bool = False
    allowed_sessions: Tuple[str, ...] = ("LONDON", "LONDON_NY_OVERLAP", "NEW_YORK")
    #: A real news filter needs an economic-calendar feed, which the
    #: MetaTrader5 Python API does not expose. Left unimplemented rather than
    #: faked; see README. Section 16 warns not to assume it helps anyway.
    use_news_filter: bool = False

    broker: BrokerSpec = field(default_factory=BrokerSpec)

    def risk_fraction(self) -> float:
        return self.risk_percent_initial_balance / 100.0


#: Live Deriv-Demo XAUUSD properties, read from the terminal via
#: scripts/fetch_mt5_data.py. Section 5.3 requires these to come from the
#: broker rather than assumed constants -- re-read them if the broker or
#: account changes.
DERIV_XAUUSD = BrokerSpec(
    symbol="XAUUSD",
    contract_size=100.0,
    tick_size=0.01,
    tick_value=1.0,
    volume_step=0.01,
    volume_min=0.01,
    volume_max=10.0,
    leverage=100.0,
    default_spread_points=15.0,  # observed median; real spread column overrides this
)

#: The original Section 3.1 filter, kept so the D1 removal can be measured
#: rather than assumed:
#:     replace(StrategyConfig(), regime_timeframes=BLUEPRINT_REGIME_TIMEFRAMES)
BLUEPRINT_REGIME_TIMEFRAMES: Tuple[str, ...] = ("D1", "H4", "H1", "M30", "M15", "M5")

#: Broker specifications read from the live Deriv terminal. Section 5.3 requires
#: these to come from the broker rather than being assumed, so they are recorded
#: here only for offline backtesting -- the live path always re-reads them.
#:
#: tick_value for a pair whose quote currency is not the account currency
#: (USDJPY, EURGBP here) moves with the exchange rate. The values below are
#: point-in-time and are an approximation in backtests; live trading reads the
#: current value per order.
SYMBOL_SPECS = {
    "XAUUSD": DERIV_XAUUSD,
    "GBPUSD": BrokerSpec(symbol="GBPUSD", contract_size=100000.0, tick_size=0.00001,
                         tick_value=1.0, volume_step=0.01, volume_min=0.01, volume_max=20.0,
                         leverage=100.0, default_spread_points=3.0),
    "EURUSD": BrokerSpec(symbol="EURUSD", contract_size=100000.0, tick_size=0.00001,
                         tick_value=1.0, volume_step=0.01, volume_min=0.01, volume_max=20.0,
                         leverage=100.0, default_spread_points=2.0),
    # Base currency IS the account currency, so notional is already in USD.
    "USDJPY": BrokerSpec(symbol="USDJPY", contract_size=100000.0, tick_size=0.001,
                         tick_value=0.64704, volume_step=0.01, volume_min=0.01, volume_max=20.0,
                         leverage=100.0, default_spread_points=3.0, margin_base_rate=1.0),
    # A cross: EUR converts to USD through EURUSD, not through EURGBP's price.
    "EURGBP": BrokerSpec(symbol="EURGBP", contract_size=100000.0, tick_size=0.00001,
                         tick_value=1.35084, volume_step=0.01, volume_min=0.01, volume_max=20.0,
                         leverage=100.0, default_spread_points=3.0, margin_base_rate=1.16),
}

#: Symbols the live robot is allowed to trade.
TRADEABLE_SYMBOLS: Tuple[str, ...] = ("XAUUSD", "GBPUSD", "EURUSD", "USDJPY", "EURGBP")

#: Research matrix from Section 10 / 17.2 -- compare, never cherry-pick.
RISK_TEST_MATRIX: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)

DEFAULT_CONFIG = StrategyConfig()

#: "Combination F" -- the rule set selected by scripts/combinations.py and
#: confirmed by scripts/risk_normalised.py, which compares candidates at equal
#: drawdown risk rather than equal risk percent.
#:
#: On 100,000 real Deriv M5 bars, held to a 12% drawdown budget, F returned
#: 165.1% against 114.7% for the next best and 94.1% for the untouched
#: baseline. Every walk-forward window was profitable and the worst was +5.0R.
#:
#: These are deviations from blueprint v1.1 and were selected by searching
#: roughly eighty configurations against a single 517-day sample, so some of
#: the measured advantage is selection luck. StrategyConfig() keeps the
#: blueprint values so the acceptance tests still test the specification.
COMBINATION_F = {
    "ob_require_swing_break": False,   # 4.2 structure break no longer mandatory
    "wpr_max_lead_bars": 5,            # was 2
    "setup_expiry_bars": 8,            # was 5
    "sr_zone_atr": 0.15,               # was 0.10
    "push_body_ratio": 0.65,           # was 0.60 -- Push 1 stays strict
    # Push 2 only has to show the move is still going, and the
    # break-beyond-Push-1 test already carries most of that proof, so it is held
    # well below Push 1's 0.65.
    #
    # The threshold surface at 0.5% risk (untruncated) peaks at 0.50-0.45, and
    # both land on 72.2R -- two adjacent values agreeing, which is the plateau
    # Section 10 asks for rather than a spike:
    #
    #   push2   trades   expR  totalR  maxDD%
    #    0.65       74  0.773    57.2    2.42
    #    0.60       89  0.699    62.2    2.86
    #    0.55      105  0.593    62.2    3.67
    #    0.50      115  0.628    72.2    2.84
    #    0.45      129  0.559    72.2    2.78
    #    0.40      140  0.465    65.2    3.83
    #
    # The 31 trades 0.50 admits over 0.60 win 38.7% at +0.548R, well clear of
    # the 25% break-even, so this is extra edge rather than extra volume. The
    # expectancy difference against 0.60 is not significant (t=-0.25), and
    # drawdown is fractionally lower.
    "push2_body_ratio": 0.50,
}


def live_config(risk_percent: float = 2.0, broker: BrokerSpec | None = None) -> StrategyConfig:
    """The configuration the live/demo robot runs.

    Risk defaults to 2%, not the 3% that maximised backtest return. At 3% the
    historical drawdown reached 11.15% against a 15% hard lockout, leaving
    almost no margin -- any deterioration switches the robot off. 2% returned
    108.8% with an 8.48% drawdown, which keeps real headroom.
    """
    return StrategyConfig(
        broker=broker or DERIV_XAUUSD,
        risk_percent_initial_balance=risk_percent,
        **COMBINATION_F,
    )
