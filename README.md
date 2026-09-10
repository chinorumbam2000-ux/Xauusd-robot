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

# 7. Rolling walk-forward windows
python -m xauusd_robot.cli walk-forward --data data/XAUUSD_M5_synthetic.csv --train-bars 60000 --test-bars 12000
```

Walk-forward scores each test window using a run that also covers its training history, then
counts only trades that *enter* inside the test window. Scoring a bare test slice would score
noise, because a D1 EMA200 needs ~57,600 M5 bars before it means anything.

Bring your own data by exporting M5 bars to CSV with `time, open, high, low, close`
(plus optional `volume` and `spread` in points). Column names are matched case-insensitively.

### Localhost dashboard

```powershell
$env:MT5_LOGIN="your_login"; $env:MT5_SERVER="Broker-Demo"; $env:MT5_PASSWORD="..."
python scripts/run_dashboard.py
# then open http://127.0.0.1:8765
```

Runs the live trader in a background thread and serves a monitoring/control UI built on the
standard library (no new dependency). It covers the blueprint's "Recommended Demo Dashboard":
six EMA200 regime lights, live price/spread/ATR, Williams %R state, the setup state machine with
its progress chips (zone reaction → WPR extreme → WPR exit → Push 1 → Push 2), active zones with
age and confluence, risk budget and sizing, every circuit breaker with manual reset buttons, the
open position with a close control, and a live event log.

The robot starts in **dry run**; arming live orders is an explicit button that refuses non-demo
accounts and refuses to arm while the terminal's AutoTrading switch is off. The server binds to
loopback only — its control endpoints mutate a live trading session, so never expose it to a
network.

Positions the robot does not own (different magic number) are listed separately, because they
still move account equity and therefore count toward the 15% peak-equity lockout.

### Real data from MetaTrader 5

`scripts/fetch_mt5_data.py` pulls real M5 history and the broker's live symbol specification
straight from a local MT5 terminal. It is **read-only** — it calls only data functions and never
places, modifies or closes an order. Credentials come from the environment so they are never
written to a file or committed:

```powershell
$env:MT5_LOGIN    = "your_login"
$env:MT5_SERVER   = "Broker-Demo"
$env:MT5_PASSWORD = "..."            # never hard-code; clear it afterwards
python scripts/fetch_mt5_data.py --years 5 --out data/XAUUSD_M5_live.csv

python -m xauusd_robot.cli backtest --data data/XAUUSD_M5_live.csv --broker deriv --risk 1.0 --balance 10000
```

If the terminal is already running and logged in, omit the credentials entirely and the script
attaches to that session. Two MT5 quirks the script works around: a single `copy_rates_range`
call stops returning data past roughly 180 days, and `copy_rates_from_pos` is capped by the
terminal's `maxbars`. It therefore walks backwards in 30-day windows and stops when the broker's
history floor announces itself.

Never send broker credentials to a third-party "MT5 REST API" service. MetaQuotes publishes no
official REST API, so those are all third parties, and handing one your login hands it account
control. The local Python integration above keeps everything on your machine.

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

## First run on real data — and why it proves nothing yet

100,000 real M5 bars from a Deriv demo account (2025-04-11 → 2026-09-10, 517 days, median
spread 15 points), $10,000 start, 1% risk:

| | |
| --- | --- |
| Trades | 15 (1.72/month) |
| Win rate | 33.3% (break-even for 1:3 is 25%) |
| Expectancy | +0.33R · profit factor 1.47 |
| Net return | +4.44% · max drawdown 3.41% |
| Worst streak | 3 consecutive losses |

**Do not read this as an edge.** Three reasons:

1. **The sample is far too small.** Fifteen trades cannot distinguish a 33% win rate from a 25%
   one; the standard error on win rate at n=15 is roughly ±12 points.
2. **The profit is two trades.** Both confluence-score-2 setups won (+6R combined); the other
   thirteen trades net **−1R**. Remove two trades and the result is a small loss.
3. **There is no out-of-sample segment.** The D1 EMA200 consumes ~57,600 bars of warm-up, so of
   100,000 bars only ~42,000 are tradeable at all. A chronological in-sample/validation/OOS split
   (Section 9.2) is not yet possible — every segment would be shorter than the warm-up.

The risk matrix does produce one actionable result. Because risk % does not change the signals,
the trade sequence is identical across settings and only the scaling differs:

| Risk | Net return | Max drawdown |
| --- | --- | --- |
| 0.5% | +2.22% | 1.54% |
| 1.0% | +4.44% | 3.41% |
| 2.0% | +9.37% | 6.50% |
| 3.0% | +14.62% | 9.39% |
| 5.0% | +24.75% | **14.69%** |

At 5%, a mere 3-loss streak drives drawdown to 14.69% against the 15% peak-equity lockout — a
fourth consecutive loss would have locked the account out entirely. That is concrete support for
Section 16's warning to treat 5% as a research variant rather than an automatic live setting.

**Next step is more history**, not more parameter tuning. Deriv's demo server caps M5 depth at
roughly 17 months; a broker or data vendor with several years of M5 gold is required before any
of the Section 9.2 validation (out-of-sample, walk-forward) can run.

## Design decisions worth knowing

**No look-ahead.** Higher-timeframe bars carry an explicit `close_time`, and each M5 bar is
joined only to higher-timeframe bars that had already closed (`merge_asof`, backward). A
still-forming H4 candle can never influence a decision. Similarly, a 2-left/2-right pivot only
becomes visible two bars after the pivot itself, and a zone is never eligible on the same bar
that confirmed it.

**Live uses the broker's own higher-timeframe bars, not resampled M5.** Resampling is wrong
twice over. The daily bar lands on UTC midnight rather than the broker's trading day, and a
200-period EMA seeded from a short resampled series stays contaminated by its seed for roughly
3x its span. Measured against this Deriv feed, resampling produced a D1 EMA200 of **4,390.69
from just 67 valid values**, against the broker's true **4,358.42** — a 31-point error, easily
enough to flip a regime verdict. The live path now pulls each timeframe natively and requests
enough history for the EMA to converge (`3 x ema_period` beyond the analysis window). Offline
CSV backtests still resample, since a single M5 file is all they have; prefer per-timeframe
exports when precision matters.

**Closed-bar verdicts differ from live price, by design.** Section 3.1 compares each timeframe's
*last closed* bar to its EMA200, so a daily candle that closed above the EMA keeps the D1 light
green all day even while price trades below it. This looks like a bug and is not one, so the
dashboard shows both the closed-bar verdict and a live-price arrow, and explicitly flags the
timeframes where they disagree.

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

## Blueprint coverage

Implemented, by section:

| Section | Item | Status |
| --- | --- | --- |
| 3.1–3.5 | Regime filter, zones, WPR, push candles, entry sequence | done |
| 4.1–4.7 | FVG / OB / S&R definitions, reaction, WPR timing, confluence | done |
| 5.1–5.5 | Risk matrix, fixed-initial-balance model, dynamic sizing, structural SL, 3R, all circuit breakers | done |
| 6.1 | Module split (MarketData … Logger) | done, as Python modules |
| 6.2 | Pandas/NumPy analysis, Matplotlib diagnostics | done |
| 7 | State machine | done |
| 9.1–9.4 | Data handling, chronological segmentation, mandatory metrics, 1:3 maths | done |
| 10 | Parameter sweeps, spread cap, plateau-not-peak selection | done |
| 11.1 | Same code path for demo/forward test, restart recovery, rejection logging | done |
| 12 | VPS deployment, state persistence, daily log rotation, alerts, dashboard | dashboard/alerts/rotation done; VPS is yours to host |
| 13 | Every configuration parameter | done |
| 14 | Log every setup, not only trades; full rule snapshot | done |
| 15 | Acceptance tests | 77 automated tests |
| 17 | Implementation priorities and definition of done | done |

Deliberately **not** implemented:

- **News filter** (Section 13, `UseNewsFilter`). A real one needs an economic-calendar feed, which
  the MetaTrader5 Python API does not expose. The flag exists; the behaviour does not. Faking it
  would be worse than omitting it, and Section 16 warns against assuming it helps at all.
- **Break-even and trailing stops** (Section 5.4). The blueprint explicitly excludes them from the
  v1.1 baseline and defers them to later variants.
- **Session filter** is implemented but **off by default** (Section 10 lists it as a later
  experiment). Enable via `use_session_filter`; `allowed_sessions` defaults to London/overlap/NY.
- **MQL5 EA** (Phases 2–3). The blueprint recommends MQL5 for live execution; this Python layer is
  the research counterpart and the reference the EA should be validated against. The live bridge
  here covers Phase 6 demo forward testing directly.

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
