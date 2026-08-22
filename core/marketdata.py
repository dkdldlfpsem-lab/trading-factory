"""Loaders for normalized market data (bar streams + macro series).

Bars are ALWAYS consumed one at a time (streaming); nothing here computes
whole-history statistics, so strategies cannot accidentally look ahead.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

NORMALIZED_DIR = Path("data/market/normalized")

# Publication lag applied to macro series (days). CPI for month M is released
# around mid M+1; 45 days after month start is a conservative no-lookahead lag.
CPI_PUBLICATION_LAG_DAYS = 45


@dataclass(frozen=True)
class Bar:
    ts: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_bars(symbol: str, timeframe: str, data_dir: Path = NORMALIZED_DIR) -> list[Bar]:
    path = data_dir / f"{symbol}_{timeframe}.csv"
    bars: list[Bar] = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            bars.append(
                Bar(
                    ts=date.fromisoformat(rec["ts"]),
                    open=float(rec["open"]),
                    high=float(rec["high"]),
                    low=float(rec["low"]),
                    close=float(rec["close"]),
                    volume=float(rec["volume"]),
                )
            )
    bars.sort(key=lambda b: b.ts)
    return bars


def iter_bars(symbol: str, timeframe: str, data_dir: Path = NORMALIZED_DIR) -> Iterator[Bar]:
    """Stream bars one at a time, oldest first."""
    yield from load_bars(symbol, timeframe, data_dir)


def load_cpi_yoy(data_dir: Path = NORMALIZED_DIR) -> list[tuple[date, float]]:
    """CPI year-over-year change as (effective_date, yoy) points.

    effective_date already includes the publication lag, so a strategy may use
    any point whose effective_date <= current bar date without lookahead.
    """
    path = data_dir / "CPI_1mo.csv"
    raw: list[tuple[date, float]] = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            raw.append((date.fromisoformat(rec["ts"]), float(rec["value"])))
    raw.sort(key=lambda x: x[0])
    by_month = {(d.year, d.month): v for d, v in raw}
    out: list[tuple[date, float]] = []
    for d, v in raw:
        prev = by_month.get((d.year - 1, d.month))
        if prev and prev > 0:
            eff = d + timedelta(days=CPI_PUBLICATION_LAG_DAYS)
            out.append((eff, v / prev - 1.0))
    return out


def load_cash_rate(data_dir: Path = NORMALIZED_DIR) -> list[tuple[date, float]]:
    """Conservative cash/T-bill proxy from the US 10y yield (Shiller).

    3-month T-bills historically sit ~1-1.5%p below the 10y yield, so we use
    max(0, 10y - 1.0%p) as annualized cash interest. Effective ~1 month after
    the observation month starts (no lookahead).
    """
    path = data_dir / "RATE_1mo.csv"
    out: list[tuple[date, float]] = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            d = date.fromisoformat(rec["ts"])
            annual = max(0.0, float(rec["value"]) - 1.0) / 100.0
            out.append((d + timedelta(days=31), annual))
    out.sort(key=lambda x: x[0])
    return out


class MacroView:
    """Streaming point-in-time access to a macro series.

    value_at(ts) returns the latest value whose effective date <= ts.
    Must be queried with non-decreasing timestamps (bar-by-bar).
    """

    def __init__(self, series: list[tuple[date, float]]):
        self._series = sorted(series, key=lambda x: x[0])
        self._idx = -1

    def value_at(self, ts: date) -> float | None:
        while self._idx + 1 < len(self._series) and self._series[self._idx + 1][0] <= ts:
            self._idx += 1
        if self._idx < 0:
            return None
        return self._series[self._idx][1]

    def prev_value(self) -> float | None:
        """Value one release before the current one (for accel/decel checks)."""
        if self._idx < 1:
            return None
        return self._series[self._idx - 1][1]
