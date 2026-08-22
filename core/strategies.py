"""Universal trend/regime strategies with streaming (incremental) indicators.

Design constraints, per research policy:
- ONE shared parameter set for every asset (BTC, S&P500, Gold, WTI, ...).
  Nothing is tuned per symbol — that would be curve fitting.
- Lookbacks are expressed in MONTHS and converted to bars per timeframe, so
  the same parameters mean the same thing on daily and monthly data.
- All indicators update one bar at a time; a strategy never sees the future.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from core.marketdata import Bar, MacroView


@dataclass(frozen=True)
class StrategyParams:
    """Shared, timeframe-agnostic parameters (lookbacks in months)."""

    mom_lookbacks_months: tuple[int, ...] = (3, 6, 12)
    regime_sma_months: int = 10  # classic 10-month SMA bull/bear split
    vol_target_annual: float = 0.15
    vol_span_months: int = 2  # EWMA span for realized vol
    max_leverage: float = 3.0
    use_regime_filter: bool = True  # longs only in bull, shorts only in bear
    allow_short: bool = True
    use_cpi_filter: bool = True
    cpi_yoy_threshold: float = 0.04  # high-inflation cutoff
    cpi_damp: float = 0.5  # exposure multiplier when inflation is high & rising

    def to_dict(self) -> dict:
        return {
            "mom_lookbacks_months": list(self.mom_lookbacks_months),
            "regime_sma_months": self.regime_sma_months,
            "vol_target_annual": self.vol_target_annual,
            "vol_span_months": self.vol_span_months,
            "max_leverage": self.max_leverage,
            "use_regime_filter": self.use_regime_filter,
            "allow_short": self.allow_short,
            "use_cpi_filter": self.use_cpi_filter,
            "cpi_yoy_threshold": self.cpi_yoy_threshold,
            "cpi_damp": self.cpi_damp,
        }


class StreamingSMA:
    def __init__(self, window: int):
        self.window = max(1, window)
        self._buf: deque[float] = deque(maxlen=self.window)
        self._sum = 0.0

    def update(self, x: float) -> float | None:
        if len(self._buf) == self._buf.maxlen:
            self._sum -= self._buf[0]
        self._buf.append(x)
        self._sum += x
        if len(self._buf) < self.window:
            return None
        return self._sum / self.window


class StreamingVol:
    """EWMA volatility of log returns, annualized."""

    def __init__(self, span_bars: int, periods_per_year: float):
        self.alpha = 2.0 / (max(2, span_bars) + 1.0)
        self.ppy = periods_per_year
        self._var: float | None = None
        self._last: float | None = None
        self.n = 0

    def update(self, close: float) -> float | None:
        if self._last is not None and self._last > 0 and close > 0:
            r = math.log(close / self._last)
            if self._var is None:
                self._var = r * r
            else:
                self._var = (1 - self.alpha) * self._var + self.alpha * r * r
            self.n += 1
        self._last = close
        if self._var is None or self.n < 5:
            return None
        return math.sqrt(self._var * self.ppy)


@dataclass
class RegimeState:
    """Shared bull/bear regime tracker (close vs long SMA)."""

    sma: StreamingSMA
    is_bull: bool | None = None

    def update(self, close: float) -> bool | None:
        s = self.sma.update(close)
        if s is not None:
            self.is_bull = close > s
        return self.is_bull


def cpi_risk_multiplier(macro: MacroView | None, bar_ts, params: StrategyParams) -> float:
    """Damp exposure when inflation is high AND accelerating (uncertainty up)."""
    if not params.use_cpi_filter or macro is None:
        return 1.0
    yoy = macro.value_at(bar_ts)
    if yoy is None:
        return 1.0
    prev = macro.prev_value()
    if yoy > params.cpi_yoy_threshold and (prev is None or yoy >= prev):
        return params.cpi_damp
    return 1.0


class TrendFollowStrategy:
    """Time-series momentum + vol targeting + regime & CPI filters.

    mode="futures": signed target in [-max_leverage, +max_leverage]
    mode="spot":    long-only target in [0, 1]
    """

    def __init__(self, params: StrategyParams, bars_per_month: int, periods_per_year: float, mode: str = "futures"):
        self.p = params
        self.mode = mode
        self.bpm = bars_per_month
        self.lookbacks = [max(1, lb * bars_per_month) for lb in params.mom_lookbacks_months]
        self._closes: deque[float] = deque(maxlen=max(self.lookbacks) + 1)
        self.regime = RegimeState(StreamingSMA(max(2, params.regime_sma_months * bars_per_month)))
        self.vol = StreamingVol(max(2, params.vol_span_months * bars_per_month), periods_per_year)

    def _trend_score(self) -> float | None:
        n = len(self._closes)
        if n < max(self.lookbacks) + 1:
            return None
        c = self._closes[-1]
        signs = []
        for lb in self.lookbacks:
            past = self._closes[n - 1 - lb]
            if past <= 0:
                return None
            signs.append(1.0 if c > past else (-1.0 if c < past else 0.0))
        return sum(signs) / len(signs)

    def on_bar(self, bar: Bar, macro: MacroView | None) -> float:
        self._closes.append(bar.close)
        is_bull = self.regime.update(bar.close)
        vol = self.vol.update(bar.close)
        trend = self._trend_score()
        if trend is None or vol is None or vol <= 0:
            return 0.0

        target = trend
        if self.p.use_regime_filter and is_bull is not None:
            if is_bull and target < 0:
                target = 0.0  # no shorts in a bull market
            elif not is_bull and target > 0:
                target = 0.0  # no longs in a bear market

        lev = min(self.p.vol_target_annual / vol, self.p.max_leverage)
        target *= lev
        target *= cpi_risk_multiplier(macro, bar.ts, self.p)

        if self.mode == "spot" or not self.p.allow_short:
            target = max(0.0, target)
        if self.mode == "spot":
            target = min(target, 1.0)
        return target
