"""Account-level risk shared across symbols.

Blueprint v1.1 is a single-instrument specification, so it has no guidance for
running several symbols at once. The rules here are therefore engineering
additions, chosen to preserve the intent of Section 5.5 rather than to extend
its letter: the circuit breakers exist to cap how much of the account can be
lost at once, and that intent breaks the moment five symbols each open a
"one position at a time" trade.

Two hazards drive the design.

Aggregate heat. Five symbols at 2% risk each is 10% of the account in flight
simultaneously. The 15% peak-equity lockout would then be reachable from a
single bad hour rather than a bad month.

Correlation. EURUSD, GBPUSD and EURGBP are close to the same position:
EURGBP is arithmetically EURUSD divided by GBPUSD, so a long EURUSD plus a
short GBPUSD is approximately a long EURGBP. Sizing them as three independent
2% bets understates the true exposure badly -- they can all lose together, and
in a dollar move they will.

So symbols are grouped into correlation clusters and only one position per
cluster is permitted by default.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

#: Symbols that move together closely enough to be treated as one exposure.
#: The USD legs dominate the majors, so they share a cluster; EURGBP sits with
#: them because it is a ratio of two of its members.
CORRELATION_CLUSTERS: Dict[str, Tuple[str, ...]] = {
    "EUR_GBP_COMPLEX": ("EURUSD", "GBPUSD", "EURGBP"),
    "USD_JPY": ("USDJPY",),
    "METALS": ("XAUUSD", "XAUUSDmicro", "XAUEUR"),
}


def cluster_of(symbol: str) -> str:
    for name, members in CORRELATION_CLUSTERS.items():
        if symbol in members:
            return name
    return f"UNGROUPED_{symbol}"


@dataclass
class PortfolioState:
    initial_balance: float
    running_equity_peak: float
    drawdown_locked: bool = False
    target_reached: bool = False
    broker_day: str = ""
    daily_realized_r: float = 0.0
    trades_today: int = 0
    losses_today: int = 0
    closed_balance: float = 0.0
    open_by_symbol: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def new(cls, balance: float) -> "PortfolioState":
        return cls(initial_balance=balance, running_equity_peak=balance, closed_balance=balance)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PortfolioState":
        return cls(**json.loads(text))


@dataclass(frozen=True)
class PortfolioLimits:
    """Account-wide caps. Defaults are deliberately tighter than per-symbol
    limits multiplied by the symbol count."""

    max_total_positions: int = 2
    max_positions_per_cluster: int = 1
    max_trades_per_day: int = 6
    max_losses_per_day: int = 3
    daily_loss_limit_r: float = -3.0
    max_peak_equity_drawdown: float = 0.15
    target_multiplier: float = 10.0


class PortfolioRisk:
    """Shared gate every symbol must pass before opening a position."""

    def __init__(self, state: PortfolioState, limits: Optional[PortfolioLimits] = None):
        self.state = state
        self.limits = limits or PortfolioLimits()

    # ------------------------------------------------------------------
    @property
    def total_open(self) -> int:
        return sum(self.state.open_by_symbol.values())

    def cluster_open(self, cluster: str) -> int:
        return sum(
            count for symbol, count in self.state.open_by_symbol.items()
            if count and cluster_of(symbol) == cluster
        )

    def can_open(self, symbol: str) -> Tuple[bool, str]:
        s, lim = self.state, self.limits
        if s.target_reached:
            return False, "portfolio_target_reached"
        if s.drawdown_locked:
            return False, "portfolio_drawdown_locked"
        if self.total_open >= lim.max_total_positions:
            return False, f"portfolio_max_positions({lim.max_total_positions})"

        cluster = cluster_of(symbol)
        if self.cluster_open(cluster) >= lim.max_positions_per_cluster:
            return False, f"correlated_position_open({cluster})"

        if s.trades_today >= lim.max_trades_per_day:
            return False, f"portfolio_max_trades_today({lim.max_trades_per_day})"
        if s.losses_today >= lim.max_losses_per_day:
            return False, f"portfolio_max_losses_today({lim.max_losses_per_day})"
        if s.daily_realized_r <= lim.daily_loss_limit_r:
            return False, f"portfolio_daily_loss_limit({lim.daily_loss_limit_r}R)"
        return True, "ok"

    # ------------------------------------------------------------------
    def register_open(self, symbol: str) -> None:
        self.state.open_by_symbol[symbol] = self.state.open_by_symbol.get(symbol, 0) + 1
        self.state.trades_today += 1

    def register_close(self, symbol: str, r_multiple: float, balance: float) -> None:
        current = self.state.open_by_symbol.get(symbol, 0)
        self.state.open_by_symbol[symbol] = max(0, current - 1)
        self.state.daily_realized_r += r_multiple
        self.state.closed_balance = balance
        if r_multiple < 0:
            self.state.losses_today += 1
        if balance >= self.state.initial_balance * self.limits.target_multiplier:
            self.state.target_reached = True

    def update_equity(self, equity: float) -> None:
        s = self.state
        s.running_equity_peak = max(s.running_equity_peak, equity)
        if s.running_equity_peak > 0:
            drawdown = 1 - equity / s.running_equity_peak
            if drawdown >= self.limits.max_peak_equity_drawdown:
                s.drawdown_locked = True

    def roll_day(self, day: str) -> None:
        if day != self.state.broker_day:
            self.state.broker_day = day
            self.state.daily_realized_r = 0.0
            self.state.trades_today = 0
            self.state.losses_today = 0

    def reset_drawdown_lock(self) -> None:
        self.state.drawdown_locked = False
        self.state.running_equity_peak = self.state.closed_balance

    def reset_target(self) -> None:
        self.state.target_reached = False

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        lim = self.limits
        return {
            "total_open": self.total_open,
            "max_total_positions": lim.max_total_positions,
            "max_positions_per_cluster": lim.max_positions_per_cluster,
            "open_by_symbol": dict(self.state.open_by_symbol),
            "clusters": {
                name: self.cluster_open(name)
                for name in {cluster_of(s) for s in self.state.open_by_symbol}
            },
            "trades_today": self.state.trades_today,
            "max_trades_per_day": lim.max_trades_per_day,
            "losses_today": self.state.losses_today,
            "max_losses_per_day": lim.max_losses_per_day,
            "daily_realized_r": round(self.state.daily_realized_r, 2),
            "daily_loss_limit_r": lim.daily_loss_limit_r,
            "drawdown_locked": self.state.drawdown_locked,
            "target_reached": self.state.target_reached,
            "equity_peak": round(self.state.running_equity_peak, 2),
            "initial_balance": round(self.state.initial_balance, 2),
        }
