"""Section 14: Logging, Analytics, and Performance Metrics.

"Every potential setup should be logged, not only executed trades. This
allows the strategy to explain why it did or did not act."

Two machine-readable streams are produced:

``events``  -- one row per setup state transition / rejection, with the
               complete rule snapshot and a deterministic reason code.
``trades``  -- one row per executed trade, with entry/exit, risk, volume,
               spread, and outcome in money, percent, and R.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class RunLogger:
    events: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    equity: List[Dict] = field(default_factory=list)

    def log_event(self, **row) -> None:
        self.events.append(row)

    def log_trade(self, **row) -> None:
        self.trades.append(row)

    def log_equity(self, **row) -> None:
        self.equity.append(row)

    # ------------------------------------------------------------------
    def events_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity)

    def write(self, out_dir: str, prefix: str = "") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for name, frame in (
            ("events", self.events_df()),
            ("trades", self.trades_df()),
            ("equity", self.equity_df()),
        ):
            path = os.path.join(out_dir, f"{prefix}{name}.csv")
            frame.to_csv(path, index=False)
            paths[name] = path
        return paths
