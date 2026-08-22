import math
from datetime import date, timedelta

from apps.worker.jobs.research_loop import segment_metrics, trend_criteria
from core.marketdata import Bar
from core.options import OptionsParams, WheelOptionsBacktester, bs_price
from core.strategies import StrategyParams, StreamingSMA, StreamingVol, TrendFollowStrategy


def test_black_scholes_put_call_parity():
    s, k, t, sigma = 100.0, 95.0, 0.25, 0.3
    call = bs_price(s, k, t, sigma, call=True)
    put = bs_price(s, k, t, sigma, call=False)
    assert abs((call - put) - (s - k)) < 1e-9  # r=0 parity: C - P = S - K


def test_black_scholes_expiry_is_intrinsic():
    assert bs_price(120.0, 100.0, 0.0, 0.3, call=True) == 20.0
    assert bs_price(80.0, 100.0, 0.0, 0.3, call=False) == 20.0


def test_streaming_sma_matches_batch():
    sma = StreamingSMA(3)
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = [sma.update(v) for v in vals]
    assert out[:2] == [None, None]
    assert out[2:] == [2.0, 3.0, 4.0]


def test_streaming_vol_positive_and_annualized():
    vol = StreamingVol(span_bars=10, periods_per_year=252)
    v = None
    px = 100.0
    for i in range(50):
        px *= 1.01 if i % 2 == 0 else 0.99
        v = vol.update(px)
    assert v is not None and v > 0.05


def _bars(prices):
    d0 = date(2020, 1, 1)
    return [Bar(d0 + timedelta(days=i), p, p, p, p, 0.0) for i, p in enumerate(prices)]


def test_trend_strategy_goes_long_in_uptrend_only():
    params = StrategyParams(mom_lookbacks_months=(1,), regime_sma_months=1, vol_span_months=1)
    strat = TrendFollowStrategy(params, bars_per_month=5, periods_per_year=252, mode="spot")
    up = [100.0 * (1.015**i) for i in range(60)]
    last = 0.0
    for b in _bars(up):
        last = strat.on_bar(b, None)
    assert last > 0

    strat2 = TrendFollowStrategy(params, bars_per_month=5, periods_per_year=252, mode="spot")
    down = [100.0 * (0.985**i) for i in range(60)]
    for b in _bars(down):
        last = strat2.on_bar(b, None)
    assert last == 0.0


def test_wheel_backtester_runs_and_sells_premium_in_bull():
    sp = StrategyParams(regime_sma_months=1, vol_span_months=1)
    op = OptionsParams()
    bt = WheelOptionsBacktester(sp, op, bars_per_month=5, periods_per_year=252)
    prices = [100.0 * (1.002**i) * (1 + (0.01 if i % 7 == 0 else -0.003)) for i in range(300)]
    res = bt.run(_bars(prices))
    assert len(res.equity_curve) == 300
    assert res.n_trades > 0
    assert res.final_equity > 0


def test_segment_metrics_known_curve():
    curve = [100.0, 110.0, 121.0]
    m = segment_metrics(curve, 0, ppy=1)
    assert abs(m["total_return"] - 0.21) < 1e-9
    assert m["max_drawdown"] == 0.0
    m2 = segment_metrics([100.0, 50.0, 75.0], 0, ppy=252)
    assert abs(m2["max_drawdown"] - 0.5) < 1e-9


def test_trend_criteria_requires_all_assets_profitable():
    def entry(tr, sh=1.0, dd=0.1):
        return {"oos": {"total_return": tr, "sharpe": sh, "max_drawdown": dd}}

    good = {
        "futures": {"A": entry(0.5), "B": entry(0.3)},
        "spot": {"A": entry(0.2), "B": entry(0.1)},
    }
    ok, _ = trend_criteria(good)
    assert ok
    bad = {
        "futures": {"A": entry(0.5), "B": entry(-0.1)},
        "spot": {"A": entry(0.2), "B": entry(0.1)},
    }
    ok, checks = trend_criteria(bad)
    assert not ok
    assert not checks["futures_all_profitable_oos"]
