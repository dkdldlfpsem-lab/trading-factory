"""Synthetic options backtest (bar-by-bar) on top of underlying price data.

We have no historical option chains, so options are priced with Black-Scholes
using trailing realized volatility times an implied-vol markup. The markup
models the variance risk premium (options usually trade above subsequently
realized vol), which is the documented, cross-asset source of edge for
premium-selling strategies. A bid/ask spread haircut is charged on every
option trade so fills are worse than mid.

Strategy: regime-filtered "wheel" — identical rules for every asset:
- Bull regime (close > long SMA): sell a ~1-month out-of-the-money
  cash-secured put. If assigned at expiry, hold the shares and sell a
  ~1-month out-of-the-money covered call until called away.
- Bear regime: buy back open options, liquidate shares, sit in cash.

No lookahead: decisions are made at bar close and executed on the NEXT bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from core.backtest import Result
from core.marketdata import Bar, MacroView
from core.strategies import RegimeState, StreamingSMA, StreamingVol, StrategyParams, cpi_risk_multiplier


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float, sigma: float, call: bool, r: float = 0.0) -> float:
    if t_years <= 0 or sigma <= 0:
        intrinsic = (spot - strike) if call else (strike - spot)
        return max(0.0, intrinsic)
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / sq
    d2 = d1 - sq
    if call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


@dataclass
class OptionsParams:
    otm_put: float = 0.05  # sell puts 5% below spot
    otm_call: float = 0.05  # sell calls 5% above spot
    expiry_days: int = 30
    iv_markup: float = 1.15  # implied = realized * markup (variance risk premium)
    spread_frac: float = 0.05  # bid/ask haircut, fraction of premium
    notional_frac: float = 1.0  # fraction of equity securing the short put

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class _ShortOption:
    is_call: bool
    strike: float
    expiry: date
    qty: float  # underlying units


class WheelOptionsBacktester:
    """Regime-filtered wheel; shares StrategyParams regime/vol settings."""

    def __init__(
        self,
        strat_params: StrategyParams,
        opt_params: OptionsParams,
        bars_per_month: int,
        periods_per_year: float,
        initial_equity: float = 100_000.0,
        cash_rate: MacroView | None = None,
    ):
        self.sp = strat_params
        self.op = opt_params
        self.regime = RegimeState(StreamingSMA(max(2, strat_params.regime_sma_months * bars_per_month)))
        self.vol = StreamingVol(max(2, strat_params.vol_span_months * bars_per_month), periods_per_year)
        self.ppy = periods_per_year
        self.initial_equity = initial_equity
        self.cash_rate = cash_rate

    def _iv(self, realized: float) -> float:
        return max(0.05, realized * self.op.iv_markup)

    def run(self, bars: list[Bar], macro: MacroView | None = None) -> Result:
        res = Result()
        cash = self.initial_equity
        shares = 0.0
        short_opt: _ShortOption | None = None
        # action decided at previous close, executed on this bar
        pending: dict | None = None
        realized_vol: float | None = None

        for bar in bars:
            px = bar.close

            # interest on cash collateral (T-bill proxy), as in the PUT index
            if self.cash_rate is not None and cash > 0:
                r = self.cash_rate.value_at(bar.ts)
                if r:
                    cash *= 1.0 + r / self.ppy

            # 0) expiry settlement (uses expiry-date price = this bar's close)
            if short_opt is not None and bar.ts >= short_opt.expiry:
                if short_opt.is_call:
                    if px > short_opt.strike:  # called away
                        cash += short_opt.strike * short_opt.qty
                        shares -= short_opt.qty
                else:
                    if px < short_opt.strike:  # assigned
                        cash -= short_opt.strike * short_opt.qty
                        shares += short_opt.qty
                short_opt = None

            # 1) execute pending decision from the previous bar
            if pending is not None and realized_vol is not None:
                act = pending["action"]
                if act == "sell_put" and short_opt is None and shares == 0:
                    strike = px * (1.0 - pending["otm"])
                    qty = (pending["risk_mult"] * self.op.notional_frac * max(cash, 0.0)) / strike
                    if qty > 0:
                        t = self.op.expiry_days / 365.0
                        prem = bs_price(px, strike, t, self._iv(realized_vol), call=False)
                        cash += prem * qty * (1.0 - self.op.spread_frac)
                        short_opt = _ShortOption(False, strike, bar.ts + timedelta(days=self.op.expiry_days), qty)
                elif act == "sell_call" and short_opt is None and shares > 0:
                    strike = px * (1.0 + pending["otm"])
                    t = self.op.expiry_days / 365.0
                    prem = bs_price(px, strike, t, self._iv(realized_vol), call=True)
                    cash += prem * shares * (1.0 - self.op.spread_frac)
                    short_opt = _ShortOption(True, strike, bar.ts + timedelta(days=self.op.expiry_days), shares)
                elif act == "liquidate":
                    if short_opt is not None:
                        t = max(0.0, (short_opt.expiry - bar.ts).days) / 365.0
                        buyback = bs_price(px, short_opt.strike, t, self._iv(realized_vol), call=short_opt.is_call)
                        cash -= buyback * short_opt.qty * (1.0 + self.op.spread_frac)
                        short_opt = None
                    if shares > 0:
                        cash += shares * px * 0.999  # small slippage on the stock leg
                        shares = 0.0
            pending = None

            # 2) update streaming state
            is_bull = self.regime.update(px)
            realized_vol = self.vol.update(px)

            # 3) mark to market
            equity = cash + shares * px
            if short_opt is not None and realized_vol is not None:
                t = max(0.0, (short_opt.expiry - bar.ts).days) / 365.0
                equity -= bs_price(px, short_opt.strike, t, self._iv(realized_vol), call=short_opt.is_call) * short_opt.qty
            res.equity_curve.append(equity)
            res.dates.append(bar.ts)
            if equity <= 0:
                break

            # 4) decide for next bar
            if is_bull is None or realized_vol is None:
                continue
            risk_mult = cpi_risk_multiplier(macro, bar.ts, self.sp)
            if not is_bull:
                if short_opt is not None or shares > 0:
                    pending = {"action": "liquidate"}
            else:
                if shares > 0 and short_opt is None:
                    pending = {"action": "sell_call", "otm": self.op.otm_call, "risk_mult": risk_mult}
                    res.n_trades += 1
                elif shares == 0 and short_opt is None and risk_mult > 0:
                    pending = {"action": "sell_put", "otm": self.op.otm_put, "risk_mult": risk_mult}
                    res.n_trades += 1

        cfg_like = _FinalizeCfg(self.ppy, self.initial_equity)
        _finalize_options(res, bars, cfg_like)
        return res


@dataclass
class _FinalizeCfg:
    periods_per_year: float
    initial_equity: float


def _finalize_options(res: Result, bars: list[Bar], cfg: _FinalizeCfg) -> None:
    curve = res.equity_curve
    if not curve:
        return
    res.final_equity = curve[-1]
    res.total_return = curve[-1] / cfg.initial_equity - 1.0
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
        if var > 0:
            res.sharpe = mean / math.sqrt(var) * math.sqrt(cfg.periods_per_year)
    peak = -math.inf
    dd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = max(dd, 1.0 - eq / peak)
    res.max_drawdown = dd
