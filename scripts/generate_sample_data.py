"""Generate SYNTHETIC XAUUSD M5 bars so the engine is runnable out of the box.

This is NOT market data. It exists only so the pipeline, tests and CLI can
be exercised end to end. Section 9.1 of the blueprint requires real,
high-quality XAUUSD history with realistic spread modelling before any
result is treated as evidence.

The generator is deliberately multi-scale, because the strategy needs all
three scales to be present before it can trade at all:

1. Trend regimes lasting days-to-weeks -- without these the six-timeframe
   EMA200 filter is MIXED essentially forever.
2. An intraday mean-reverting swing -- this is what pulls Williams %R(49)
   into its extreme while the higher timeframes stay aligned.
3. Realistic candle bodies -- body/range is drawn from a Beta distribution
   so the 60% push-candle rule sees a realistic mix of strong bars and dojis.

Note the warm-up cost of the D1 EMA200: 200 daily bars is roughly 57,600
M5 bars, so anything shorter than that produces no tradeable regime at all.
Generate at least ~300,000 bars (about three years) for a usable run.

    python scripts/generate_sample_data.py --bars 315000 --out data/XAUUSD_M5_synthetic.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def generate(bars: int = 315000, start: str = "2022-01-01", start_price: float = 1900.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 1. Trend regimes lasting days to weeks.
    drift = np.zeros(bars)
    cursor = 0
    while cursor < bars:
        length = int(rng.integers(2500, 12000))
        drift[cursor : cursor + length] = rng.normal(0, 1.0) * 0.012
        cursor += length

    # 2. Intraday mean-reverting swing (period ~1 day = 288 M5 bars).
    t = np.arange(bars)
    amplitude = 5.0 + 4.0 * np.sin(2 * np.pi * t / (288 * 9)) ** 2
    intraday = amplitude * np.sin(2 * np.pi * t / 288)
    intraday_step = np.diff(intraday, prepend=intraday[0])

    noise = rng.normal(0, 0.45, bars)
    close = start_price + np.cumsum(drift + intraday_step + noise)

    open_ = np.empty(bars)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # 3. Candle shape drawn from a Beta so body/range is realistic.
    body = np.abs(close - open_)
    body_fraction = np.clip(rng.beta(5.0, 4.0, bars), 0.05, 0.97)
    bar_range = np.maximum(body / body_fraction, body + 0.02)
    wick_total = bar_range - body
    upper_share = rng.uniform(0, 1, bars)
    high = np.maximum(open_, close) + wick_total * upper_share
    low = np.minimum(open_, close) - wick_total * (1 - upper_share)

    index = pd.date_range(start=start, periods=bars, freq="5min")
    return pd.DataFrame(
        {
            "time": index,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": rng.integers(50, 900, bars),
            "spread": np.clip(rng.normal(20, 6, bars), 8, 90).round().astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=315000, help="~315,000 = 3 years of M5")
    parser.add_argument("--out", default="data/XAUUSD_M5_synthetic.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    frame = generate(args.bars, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame):,} synthetic M5 bars to {args.out}")
    if args.bars < 60000:
        print("WARNING: under ~57,600 bars the D1 EMA200 never warms up -- no trades are possible.")
    print("SYNTHETIC DATA -- not market data. Do not treat results as evidence.")


if __name__ == "__main__":
    main()
