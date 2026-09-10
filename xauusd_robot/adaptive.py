"""Live performance tracking and edge-decay detection.

What this module does NOT do is auto-tune parameters from live results, and
that is a deliberate design decision rather than an omission.

The arithmetic: this strategy produces roughly 4-5 trades a month. The
standard error of a win rate measured over n trades is sqrt(p(1-p)/n), so at
the backtested 44% win rate:

    n =  10 trades  ->  +/- 15.4 points
    n =  25 trades  ->  +/-  9.7 points
    n =  50 trades  ->  +/-  6.9 points
    n = 100 trades  ->  +/-  4.9 points
    n = 400 trades  ->  +/-  2.4 points

A month of live trading is about 5 trades, where the confidence interval is
roughly +/- 22 points -- wider than the entire distance between a profitable
system and a losing one. Re-fitting parameters on that is fitting noise, and
would do it repeatedly, compounding the error. Blueprint Section 16 reaches
the same conclusion from the other direction: establish a deterministic
baseline first, and only add learning for a clearly defined, measurable
prediction problem.

So "continuously learn and improve" is implemented here as:

* measure every live trade against the distribution the backtest predicted,
* report honest confidence intervals rather than point estimates,
* raise an alert when live results fall statistically below expectation
  (edge decay is real and detectable, unlike per-trade improvement),
* accumulate a persistent, fully-featured trade ledger, and
* say exactly how many more trades are needed before a re-optimisation on
  live data would mean anything.

When that threshold is reached, `python -m xauusd_robot.cli reoptimise` re-runs
the sweep including live trades and reports what it finds. A human applies it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Expectation:
    """What the backtest predicted, for live results to be measured against."""

    win_rate: float          # percent
    expectancy_r: float
    sample_trades: int
    label: str = "backtest"


#: Combination F on 100,000 real Deriv M5 bars.
DEFAULT_EXPECTATION = Expectation(win_rate=44.0, expectancy_r=0.750, sample_trades=75, label="combination F backtest")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval -- correct for small samples, unlike normal approx."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def trades_for_precision(p: float, margin_points: float, z: float = 1.96) -> int:
    """Trades needed to measure a win rate to +/- margin_points percent."""
    margin = margin_points / 100.0
    if margin <= 0:
        return 0
    return int(math.ceil(z * z * p * (1 - p) / (margin * margin)))


@dataclass
class TradeLedger:
    """Persistent record of live trades, the raw material for any future re-fit."""

    path: str
    trades: List[Dict] = field(default_factory=list)

    def load(self) -> "TradeLedger":
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    self.trades = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self.trades = []
        return self

    def append(self, trade: Dict) -> None:
        self.trades.append(trade)
        self.save()

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.trades, fh, indent=2, default=str)
        except OSError:
            pass  # persistence must never take the trading loop down

    @property
    def r_values(self) -> List[float]:
        return [t["r_multiple"] for t in self.trades if t.get("r_multiple") is not None]


class PerformanceTracker:
    """Compares live results with the backtested expectation, honestly."""

    #: Below this, no statistical claim is made at all.
    MIN_TRADES_FOR_SIGNAL = 25
    #: Re-optimising on fewer than this is curve-fitting.
    MIN_TRADES_FOR_REFIT = 100

    def __init__(self, ledger: TradeLedger, expectation: Expectation = DEFAULT_EXPECTATION):
        self.ledger = ledger
        self.expectation = expectation

    # ------------------------------------------------------------------
    def assess(self) -> Dict:
        r_values = self.ledger.r_values
        n = len(r_values)
        expected = self.expectation

        report: Dict = {
            "live_trades": n,
            "expected_win_rate": expected.win_rate,
            "expected_expectancy_r": expected.expectancy_r,
            "expectation_source": expected.label,
            "min_trades_for_signal": self.MIN_TRADES_FOR_SIGNAL,
            "min_trades_for_refit": self.MIN_TRADES_FOR_REFIT,
            "trades_until_signal": max(0, self.MIN_TRADES_FOR_SIGNAL - n),
            "trades_until_refit": max(0, self.MIN_TRADES_FOR_REFIT - n),
            "auto_tuning": "disabled by design -- see xauusd_robot/adaptive.py",
        }

        if n == 0:
            report.update(verdict="NO_DATA",
                          detail="No live trades yet. Nothing can be concluded.")
            return report

        wins = sum(1 for r in r_values if r > 0)
        live_win_rate = wins / n * 100
        live_expectancy = sum(r_values) / n
        lo, hi = wilson_interval(wins, n)

        report.update({
            "live_win_rate": round(live_win_rate, 1),
            "live_expectancy_r": round(live_expectancy, 3),
            "live_total_r": round(sum(r_values), 2),
            "win_rate_ci_95": [round(lo * 100, 1), round(hi * 100, 1)],
            "ci_width_points": round((hi - lo) * 100, 1),
            "break_even_win_rate": 25.0,
        })

        # Is the backtested win rate still plausible given live results?
        expected_inside = lo * 100 <= expected.win_rate <= hi * 100
        # Is the system still plausibly above the 1:3 break-even?
        above_break_even = lo * 100 > 25.0
        below_break_even = hi * 100 < 25.0

        if n < self.MIN_TRADES_FOR_SIGNAL:
            report.update(
                verdict="INSUFFICIENT_DATA",
                detail=(f"{n} trades gives a 95% interval of "
                        f"{report['win_rate_ci_95'][0]}-{report['win_rate_ci_95'][1]}% "
                        f"({report['ci_width_points']} points wide). Too wide to act on. "
                        f"{report['trades_until_signal']} more trades before any read is meaningful."),
            )
        elif below_break_even:
            report.update(
                verdict="EDGE_LOST",
                detail=("Live win rate is significantly BELOW the 25% break-even for a 1:3 "
                        "payoff. This is decay, not variance. Stop and re-validate."),
            )
        elif not expected_inside and live_win_rate < expected.win_rate:
            report.update(
                verdict="UNDERPERFORMING",
                detail=(f"Backtested {expected.win_rate}% sits outside the live 95% interval. "
                        "Live is materially worse than predicted -- likely regime change or "
                        "an overfitted backtest. Review before adding risk."),
            )
        elif above_break_even:
            report.update(
                verdict="ON_TRACK",
                detail=("Live results remain consistent with the backtest and significantly "
                        "above break-even."),
            )
        else:
            report.update(
                verdict="CONSISTENT",
                detail=("Live results are consistent with the backtest, but the sample cannot "
                        "yet separate the system from break-even."),
            )
        return report

    # ------------------------------------------------------------------
    def precision_table(self) -> List[Dict]:
        """How much more data buys how much more certainty."""
        p = self.expectation.win_rate / 100.0
        rows = []
        for n in (10, 25, 50, 100, 200, 400):
            margin = 1.96 * math.sqrt(p * (1 - p) / n) * 100
            rows.append({
                "trades": n,
                "win_rate_margin_points": round(margin, 1),
                "months_at_5_per_month": round(n / 5.0, 1),
            })
        return rows
