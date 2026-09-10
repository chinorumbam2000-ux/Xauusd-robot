"""Section 4: Finalized Zone, Reaction, and Confirmation Rules.

Implements the three alternative M5 reaction-zone types -- Fair Value Gap
(4.1), Order Block (4.2), Support/Resistance (4.3) -- plus the shared
reaction (4.4) and invalidation/aging/first-reaction-only lifecycle rules.

``ZoneEngine`` is stateful and processes bars strictly in order (bar ``i``
is only processed once bars ``0..i`` are known), which is how all of the
lifecycle rules (touch windows, displacement windows, aging, first-
reaction-only retirement) are naturally expressed without look-ahead.

Boundary convention used throughout (derived from Section 4.1/4.2/4.4):
a BUY (bullish/support-type) zone's *near* boundary -- the one price must
close back beyond to confirm a reaction -- is its ``high``; its *far*
(invalidation) boundary is its ``low``. A SELL (bearish/resistance-type)
zone is the mirror image: near = ``low``, far = ``high``.
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ZoneType(str, enum.Enum):
    OB = "OB"
    FVG = "FVG"
    SR = "SR"


@dataclass
class Zone:
    id: int
    type: ZoneType
    direction: str  # "BUY" or "SELL"
    low: float
    high: float
    created_bar: int
    created_time: pd.Timestamp
    expires_bar: int
    touched: bool = False
    touch_bar: Optional[int] = None
    retired: bool = False
    retired_reason: Optional[str] = None

    @property
    def near_boundary(self) -> float:
        return self.high if self.direction == "BUY" else self.low

    @property
    def far_boundary(self) -> float:
        return self.low if self.direction == "BUY" else self.high

    def overlaps(self, other: "Zone") -> bool:
        return not (self.high < other.low or self.low > other.high)


@dataclass
class ReactionEvent:
    zone: Zone
    direction: str
    reaction_bar: int
    reaction_time: pd.Timestamp
    touch_bar: int
    confluence_score: int
    confluence_types: List[str]


class ZoneEngine:
    def __init__(self, config, bars: pd.DataFrame):
        self.config = config
        self.update_bars(bars)

        self.active_zones: List[Zone] = []
        self.all_zones: List[Zone] = []
        self._next_id = itertools.count(1)
        self._ob_candidates: Dict[str, int] = {}  # {"bull"|"bear": candidate bar index}

    def update_bars(self, bars: pd.DataFrame) -> None:
        """Refresh cached arrays without disturbing zone lifecycle state.

        Live trading appends one closed bar at a time; zones hold integer bar
        indices, so bars may only ever be appended, never trimmed from the
        front. Pivots are recomputed because the newest bars can confirm a
        pivot that was still pending.
        """
        self.index = bars.index
        self.open = bars["open"].to_numpy()
        self.high = bars["high"].to_numpy()
        self.low = bars["low"].to_numpy()
        self.close = bars["close"].to_numpy()
        self.atr = bars["atr"].to_numpy()
        self.bullish = bars["bullish"].to_numpy()
        self.bearish = bars["bearish"].to_numpy()

        from .structure import confirmed_pivots, last_confirmed_value

        cfg = self.config
        pivot_high, pivot_low = confirmed_pivots(
            bars["high"], bars["low"], cfg.sr_pivot_left, cfg.sr_pivot_right
        )
        self.pivot_high = pivot_high.to_numpy()
        self.pivot_low = pivot_low.to_numpy()
        self.swing_high_asof = last_confirmed_value(
            pivot_high, bars["high"], cfg.sr_pivot_right
        ).to_numpy()
        self.swing_low_asof = last_confirmed_value(
            pivot_low, bars["low"], cfg.sr_pivot_right
        ).to_numpy()

    def _new_zone(self, ztype: ZoneType, direction: str, low: float, high: float, created_bar: int) -> Zone:
        z = Zone(
            id=next(self._next_id),
            type=ztype,
            direction=direction,
            low=low,
            high=high,
            created_bar=created_bar,
            created_time=self.index[created_bar],
            expires_bar=created_bar + self.config.zone_max_age_bars,
        )
        self.active_zones.append(z)
        self.all_zones.append(z)
        return z

    # ------------------------------------------------------------------
    def process_bar(self, i: int) -> List[ReactionEvent]:
        if i < 2:
            return []
        self._expire_untouched(i)
        self._detect_fvg(i)
        self._detect_sr(i)
        self._update_ob_candidates(i)
        events = self._process_touch_and_reaction(i)
        self.active_zones = [z for z in self.active_zones if not z.retired]
        return events

    # ------------------------------------------------------------------
    def _expire_untouched(self, i: int) -> None:
        for z in self.active_zones:
            if not z.touched and not z.retired and i >= z.expires_bar:
                z.retired = True
                z.retired_reason = "expired"

    def _detect_fvg(self, i: int) -> None:
        c1_high, c1_low = self.high[i - 2], self.low[i - 2]
        c3_high, c3_low = self.high[i], self.low[i]
        min_size = self.config.fvg_min_atr * self.atr[i]
        if np.isnan(min_size):
            return
        if c3_low > c1_high and (c3_low - c1_high) >= min_size:
            self._new_zone(ZoneType.FVG, "BUY", low=c1_high, high=c3_low, created_bar=i)
        elif c3_high < c1_low and (c1_low - c3_high) >= min_size:
            self._new_zone(ZoneType.FVG, "SELL", low=c3_high, high=c1_low, created_bar=i)

    def _detect_sr(self, i: int) -> None:
        right = self.config.sr_pivot_right
        p = i - right
        if p < self.config.sr_pivot_left:
            return
        zone_half = self.config.sr_zone_atr * self.atr[i]
        if np.isnan(zone_half):
            return
        if self.pivot_high[p]:
            price = self.high[p]
            self._new_zone(ZoneType.SR, "SELL", low=price - zone_half, high=price + zone_half, created_bar=i)
        if self.pivot_low[p]:
            price = self.low[p]
            self._new_zone(ZoneType.SR, "BUY", low=price - zone_half, high=price + zone_half, created_bar=i)

    def _update_ob_candidates(self, i: int) -> None:
        """Section 4.2: confirm an OB when displacement + structure break land.

        Only the LAST opposite-colour candle before the displacement can be
        the Order Block, so at most one candidate of each polarity is kept --
        a newer opposite-colour candle replaces the older one.
        """
        cfg = self.config
        # Evaluate existing candidates against the newly closed bar first: bar
        # i can be the displacement bar for a candidate AND a fresh candidate
        # of the opposite polarity.
        for kind in ("bull", "bear"):
            cand_bar = self._ob_candidates.get(kind)
            if cand_bar is None or cand_bar >= i:
                continue
            if i - cand_bar > cfg.ob_displacement_bars:
                del self._ob_candidates[kind]
                continue
            atr_ref = self.atr[cand_bar]
            if np.isnan(atr_ref):
                continue
            if kind == "bull":
                displacement = self.close[i] - self.close[cand_bar]
                swing_ok = (not cfg.ob_require_swing_break) or (
                    not np.isnan(self.swing_high_asof[i]) and self.close[i] > self.swing_high_asof[i]
                )
                direction = "BUY"
            else:
                displacement = self.close[cand_bar] - self.close[i]
                swing_ok = (not cfg.ob_require_swing_break) or (
                    not np.isnan(self.swing_low_asof[i]) and self.close[i] < self.swing_low_asof[i]
                )
                direction = "SELL"

            if displacement >= cfg.ob_displacement_atr * atr_ref and swing_ok:
                self._new_zone(
                    ZoneType.OB, direction,
                    low=self.low[cand_bar], high=self.high[cand_bar],
                    created_bar=i,
                )
                del self._ob_candidates[kind]

        if self.bearish[i]:
            self._ob_candidates["bull"] = i  # last bearish candle -> potential bullish OB
        if self.bullish[i]:
            self._ob_candidates["bear"] = i  # last bullish candle -> potential bearish OB

    def _process_touch_and_reaction(self, i: int) -> List[ReactionEvent]:
        cfg = self.config
        events: List[ReactionEvent] = []
        for z in self.active_zones:
            if z.retired or z.created_bar >= i:
                # A zone is only live from the bar after the one that confirmed
                # it -- the confirming candle cannot also react from it.
                continue
            # General invalidation (Section 4.1/4.2): a close beyond the far
            # boundary invalidates the zone, touched or not.
            if z.direction == "BUY" and self.close[i] < z.far_boundary:
                z.retired = True
                z.retired_reason = "invalidated"
                continue
            if z.direction == "SELL" and self.close[i] > z.far_boundary:
                z.retired = True
                z.retired_reason = "invalidated"
                continue

            if not z.touched:
                if self.high[i] >= z.low and self.low[i] <= z.high:
                    z.touched = True
                    z.touch_bar = i
                else:
                    continue

            if z.touch_bar is None:
                continue
            window_end = z.touch_bar + cfg.reaction_max_bars
            if i > window_end:
                # reaction window expired without a confirmed reaction; allow re-touch later
                z.touched = False
                z.touch_bar = None
                continue

            reacted = (z.direction == "BUY" and self.close[i] > z.near_boundary) or (
                z.direction == "SELL" and self.close[i] < z.near_boundary
            )
            if reacted:
                score, types = self._confluence(z)
                if cfg.first_reaction_only:
                    z.retired = True
                    z.retired_reason = "reacted"
                else:
                    # Zone stays eligible; release the touch so it can react again.
                    z.touched = False
                    z.touch_bar = None
                events.append(
                    ReactionEvent(
                        zone=z,
                        direction=z.direction,
                        reaction_bar=i,
                        reaction_time=self.index[i],
                        touch_bar=z.touch_bar,
                        confluence_score=score,
                        confluence_types=types,
                    )
                )
        return events

    def _confluence(self, zone: Zone) -> tuple:
        types = {zone.type.value}
        for other in self.active_zones:
            if other.id == zone.id or other.retired:
                continue
            if other.direction == zone.direction and other.overlaps(zone):
                types.add(other.type.value)
        return min(len(types), 3), sorted(types)
