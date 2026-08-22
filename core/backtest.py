"""Event-driven, bar-by-bar backtest engine.

Contract (no lookahead, by construction):
- The engine feeds bars to the strategy ONE AT A TIME, oldest first.
- At each bar the strategy sees only that bar (and whatever incremental state
  it built from earlier bars) and returns a TARGET exposure as a fraction of
  current equity (signed; negative = short).
- The target decided on bar[i] is executed at bar[i+1].open — a signal
  computed from a close can never be filled at that same close.
- Trading costs (commission + slippage) are charged on traded notional; an
  annual holding cost approximates futures funding/borrow on open exposure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from core.marketdata import Bar, MacroView


class Strategy(Protocol):
    def on_bar(self, bar: Bar, macro: MacroView | None) -> float:
        """Return desired exposure (fraction of equity, signed) for the NEXT bar."""
        ...


@dataclass
class EngineConfig:
    mode: str = "futures"  # "spot" (long-only, <=1x) or "futures" (long/short, leveraged)
    cost_bps: float = 10.0  # per side, on traded notional (commission + slippage)
    holding_cost_annual: float = 0.01  # funding/borrow drag on |exposure| (futures)
    max_leverage: float = 3.0
    periods_per_year: float = 252.0
    initial_equity: float = 100_000.0


@dataclass
class Result:
    equity_curve: list[float] = field(default_factory=list)
    dates: list[object] = field(default_factory=list)
    n_trades: int = 0
    turnover: float = 0.0  # total traded notional / initial equity
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_exposure: float = 0.0
    buy_hold_return: float = 0.0

    def summary(self) -> dict:
        return {
            "final_equity": round(self.final_equity, 2),
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_trades": self.n_trades,
            "turnover": round(self.turnover, 1),
            "avg_exposure": round(self.avg_exposure, 3),
            "buy_hold_return": round(self.buy_hold_return, 4),
        }


def _clip_target(target: float, cfg: EngineConfig) -> float:
    if cfg.mode == "spot":
        return min(max(target, 0.0), 1.0)
    lev = cfg.max_leverage
    return min(max(target, -lev), lev)


def run_backtest(bars: list[Bar], strategy: Strategy, cfg: EngineConfig, macro: MacroView | None = None) -> Result:
    res = Result()
    cash = cfg.initial_equity
    units = 0.0  # asset units held (negative = short)
    pending_target: float | None = None  # decided at previous close
    traded_notional = 0.0
    exposure_sum = 0.0
    min_trade_frac = 0.005  # skip dust rebalances (<0.5% of equity)

    for bar in bars:
        # 1) execute the target decided at the previous bar, at this bar's open
        if pending_target is not None:
            price = bar.open
            equity = cash + units * price
            if equity > 0:
                target_units = pending_target * equity / price
                delta = target_units - units
                notional = abs(delta) * price
                if notional > min_trade_frac * equity:
                    cost = notional * cfg.cost_bps / 1e4
                    cash -= delta * price + cost
                    units = target_units
                    traded_notional += notional
                    res.n_trades += 1

        # 2) holding cost on open exposure (funding / borrow)
        if units != 0.0 and cfg.holding_cost_annual > 0:
            cash -= abs(units) * bar.close * cfg.holding_cost_annual / cfg.periods_per_year

        # 3) mark to market at close, record, then let the strategy decide
        equity_close = cash + units * bar.close
        res.equity_curve.append(equity_close)
        res.dates.append(bar.ts)
        if equity_close <= 0:  # blown up; halt
            break
        exposure_sum += abs(units) * bar.close / equity_close
        pending_target = _clip_target(strategy.on_bar(bar, macro), cfg)

    _finalize(res, bars, cfg, traded_notional, exposure_sum)
    return res


def _finalize(res: Result, bars: list[Bar], cfg: EngineConfig, traded_notional: float, exposure_sum: float) -> None:
    curve = res.equity_curve
    if not curve:
        return
    res.final_equity = curve[-1]
    res.total_return = curve[-1] / cfg.initial_equity - 1.0
    res.turnover = traded_notional / cfg.initial_equity
    res.avg_exposure = exposure_sum / len(curve)
    if bars:
        res.buy_hold_return = bars[len(curve) - 1].close / bars[0].close - 1.0

    years = len(curve) / cfg.periods_per_year
    if years > 0 and curve[-1] > 0:
        res.cagr = (curve[-1] / cfg.initial_equity) ** (1.0 / years) - 1.0

    rets = []
    prev = cfg.initial_equity
    for eq in curve:
        if prev > 0:
            rets.append(eq / prev - 1.0)
        prev = eq
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std > 0:
            res.sharpe = mean / std * math.sqrt(cfg.periods_per_year)

    peak = -math.inf
    max_dd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)
    res.max_drawdown = max_dd
