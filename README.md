# XAUUSD Trading Robot — Strategy Specification v1.1

A deterministic, rule-based XAUUSD (gold) trading system implemented in Python from
`Forex_Robot_XAUUSD_Blueprint_v1.1.docx`. Every rule in Sections 3–5 of that blueprint is
implemented explicitly and covered by an automated test, and every decision the engine
makes — including every rejection — is logged with a machine-readable reason code.

**This is a research and engineering implementation, not a claim of profitability.**
The strategy must prove itself through out-of-sample, walk-forward and forward testing
before any meaningful capital is used.

---

## The strategy in one paragraph

Direction is decided by a strict six-timeframe EMA200 regime filter (D1, H4, H1, M30, M15, M5 —
all six closed bars must be on the same side of their EMA200, otherwise no trade). Execution is
M5-only. The M5 zone engine tracks three *alternative* reaction zones — Order Block, Fair Value
Gap, and pivot-based Support/Resistance. When price reacts out of one of them, a setup arms for
exactly 5 bars, during which Williams %R(49) must exit an extreme and two consecutive push
candles (body/range ≥ 60%, second closing beyond the first) must form. Only then is a trade
sized: the stop is structural, and the lot size is derived from that stop distance so the money
at risk equals a configured percentage of the **initial** account balance. The target is exactly
3R. A safety layer enforces one position at a time, 3 trades/day, 2 losses/day, a −2R daily
limit, a 15% peak-equity lockout, a 3-bar cooldown and a 10× closed-balance kill switch.

## Architecture

| Module | Blueprint section | Responsibility |
| --- | --- | --- |
| `config.py` | 13 | Every parameter, frozen at the v1.1 baseline values |
| `indicators.py` | 3.1, 3.3 | EMA, Wilder ATR, Williams %R, body/range ratio |
| `data.py` | 2, 9.1 | Loading, resampling, and no-look-ahead timeframe alignment |
| `regime.py` | 3.1 | Six-timeframe EMA200 BUY / SELL / MIXED filter |
| `structure.py` | 4.3 | Confirmed 2-left/2-right swing pivots |
| `zones.py` | 4.1–4.4, 4.7 | OB / FVG / S&R detection, aging, invalidation, reaction, confluence |
| `candles.py` | 3.4, 4.6 | Two consecutive push-candle rules |
| `state_machine.py` | 7 | Armed-setup lifecycle: expiry timer, WPR sequencing, push chain |
| `risk.py` | 5 | Structural stop, risk budget, broker-normalised lots, exact 3R target |
| `safety.py` | 5.5, 12 | Circuit breakers, daily limits, cooldown, restart-safe state |
| `backtester.py` | 3.5, 9 | Bar-by-bar harness with a realistic bid/ask execution model |
| `metrics.py` | 9.3, 9.4 | Mandatory performance metrics + Monte Carlo sequence risk |
| `validation.py` | 9.2, 10 | Chronological splits, walk-forward windows, parameter sweeps |
| `logging_engine.py` | 14 | Event / trade / equity logs |
| `reporting.py` | 6.2 | Equity, drawdown, R-distribution and zone-performance charts |

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic demo data (NOT market data - see the warning below)
python scripts/generate_sample_data.py --bars 315000 --out data/XAUUSD_M5_synthetic.csv

# 2. Run the baseline backtest with a full audit trail
python -m xauusd_robot.cli backtest --data data/XAUUSD_M5_synthetic.csv --risk 1.0 --out results/baseline --chart

# 3. Compare the risk research matrix (0.5 / 1 / 2 / 3 / 5%)
python -m xauusd_robot.cli risk-matrix --data data/XAUUSD_M5_synthetic.csv

# 4. Test parameter stability - look for a plateau, not a peak
python -m xauusd_robot.cli sweep --data data/XAUUSD_M5_synthetic.csv --param setup_expiry_bars --values 3,5,8,10

# 5. Stress worse execution assumptions
python -m xauusd_robot.cli spread-stress --data data/XAUUSD_M5_synthetic.csv

# 6. In-sample / validation / out-of-sample, strictly chronological
python -m xauusd_robot.cli split --data data/XAUUSD_M5_synthetic.csv --risk 1.0
```

Bring your own data by exporting M5 bars to CSV with `time, open, high, low, close`
(plus optional `volume` and `spread` in points). Column names are matched case-insensitively.

## The setup funnel

Because the rule stack is highly selective, the most useful diagnostic is *where setups die*.
Every run reports it:

```
bars                                315,000
regime_BUY                           52,923
regime_SELL                          42,009
regime_MIXED                        220,068     <- six-TF alignment is rare
zone_reactions                      110,020
zone_reactions_regime_matched        19,377
setups_armed                         10,865
setups_reached_wpr_extreme            2,042
setups_reached_wpr_confirmed            866     <- WPR(49) timing is the first big gate
setups_entry_evaluated                   82
orders_placed                             5     <- the spread filter is the second big gate
rejected_spread_vs_atr                   48
rejected_spread_vs_sl                    29
```

Section 16 of the blueprint anticipates exactly this ("Over-defining confluence ... could starve
the system of trades"). The funnel turns that risk into a measurement rather than an opinion,
and points at which rule family to vary first during robustness testing.

## Design decisions worth knowing

**No look-ahead.** Higher-timeframe bars carry an explicit `close_time`, and each M5 bar is
joined only to higher-timeframe bars that had already closed (`merge_asof`, backward). A
still-forming H4 candle can never influence a decision. Similarly, a 2-left/2-right pivot only
becomes visible two bars after the pivot itself, and a zone is never eligible on the same bar
that confirmed it.

**Execution model.** Input prices are treated as bid. A BUY fills at ask and exits at bid; a SELL
fills at bid and exits at ask, so the spread is charged exactly once per round trip. The stop
distance used for sizing is measured from the real fill price to the real stop level, which makes
the configured risk budget the true worst-case loss. Where a bar contains both the stop and the
target, the stop is assumed to fill first; a bar that gaps past a level fills at the open.

**Warm-up.** A D1 EMA200 needs 200 daily bars — roughly 57,600 M5 bars. Anything shorter produces
no tradeable regime at all. Use at least three years of M5 history.

**Risk is fixed to the initial balance.** $50 on a $1,000 start stays $50 as the account grows,
so the *effective* risk decays (5% → 2.5% at $2,000 → 0.5% at $10,000), exactly as Section 5.2
specifies.

## Testing

```bash
python -m pytest -q     # 77 tests
```

The suite mirrors the Section 15 acceptance table: EMA alignment and no-look-ahead, FVG/OB/S&R
construction and invalidation, wick-vs-close invalidation, zone aging, first-reaction-only,
reaction timing windows, WPR lead-bar and exit sequencing, push-candle consecutiveness and
structure break, setup expiry, structural stops, risk-cap rounding, minimum-lot rejection, the
1:3 target, every circuit breaker, and restart persistence — plus end-to-end invariant checks
that no completed run ever breaches the risk cap, the daily limits, the one-position rule or the
cooldown.

## Synthetic data warning

`scripts/generate_sample_data.py` produces a multi-scale synthetic series (week-long trend
regimes + intraday mean reversion + Beta-distributed candle bodies) purely so the pipeline is
runnable and testable end to end. **It is not market data and results from it are not evidence.**
Section 9.1 requires real XAUUSD history with realistic spread modelling, covering trending,
ranging, high- and low-volatility and crisis periods.

## Status against the blueprint roadmap

- **Phase 0 — Specification frozen:** done (the blueprint).
- **Phase 1 — Research prototype:** this repository. All v1.1 rules implemented deterministically.
- **Phase 4 — Backtest harness:** done. Repeatable runs, standardised trade/signal/equity logs.
- **Phase 5 — Robust validation:** tooling present (chronological splits, walk-forward windows,
  parameter sweeps, spread stress, Monte Carlo). Needs real data to produce meaningful results.
- **Phases 2/3 — MQL5 signal + execution EA:** not implemented here. The blueprint recommends
  MQL5 for live execution; this Python layer is the research/validation counterpart and the
  reference implementation the EA should be validated against.
- **Phases 6–8 — Demo, live pilot, monitoring:** gated behind acceptance criteria, per Section 11.2.

## Licence

Provided as-is for research and educational use. Trading leveraged instruments carries
substantial risk of loss.
