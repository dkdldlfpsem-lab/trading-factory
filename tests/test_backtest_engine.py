from datetime import date, timedelta

from core.backtest import EngineConfig, run_backtest
from core.marketdata import Bar, MacroView


def make_bars(prices, opens=None):
    bars = []
    d0 = date(2020, 1, 1)
    for i, c in enumerate(prices):
        o = opens[i] if opens else c
        bars.append(Bar(ts=d0 + timedelta(days=i), open=o, high=max(o, c), low=min(o, c), close=c, volume=0.0))
    return bars


class AlwaysLong:
    def on_bar(self, bar, macro):
        return 1.0


class Recorder:
    def __init__(self):
        self.seen = []

    def on_bar(self, bar, macro):
        self.seen.append(bar.ts)
        return 0.0


def test_signal_fills_at_next_bar_open():
    # close of bar0 = 100 triggers the signal; fill must be at bar1 open = 110
    bars = make_bars([100.0, 111.0, 111.0], opens=[100.0, 110.0, 111.0])
    cfg = EngineConfig(mode="spot", cost_bps=0.0, holding_cost_annual=0.0, initial_equity=1000.0)
    res = run_backtest(bars, AlwaysLong(), cfg)
    # bought at 110 with 1000 -> 9.0909 units; equity at close 111
    assert abs(res.equity_curve[1] - 1000.0 / 110.0 * 111.0) < 1e-6


def test_bars_arrive_one_at_a_time_in_order():
    bars = make_bars([1, 2, 3, 4, 5])
    rec = Recorder()
    run_backtest(bars, rec, EngineConfig())
    assert rec.seen == [b.ts for b in bars]


def test_costs_reduce_equity():
    bars = make_bars([100.0] * 50)
    cfg_free = EngineConfig(mode="spot", cost_bps=0.0, holding_cost_annual=0.0)
    cfg_paid = EngineConfig(mode="spot", cost_bps=50.0, holding_cost_annual=0.0)
    free = run_backtest(bars, AlwaysLong(), cfg_free)
    paid = run_backtest(bars, AlwaysLong(), cfg_paid)
    assert paid.final_equity < free.final_equity


def test_spot_mode_clips_leverage_and_shorts():
    class Wild:
        def on_bar(self, bar, macro):
            return -5.0

    bars = make_bars([100.0, 90.0, 80.0, 70.0])
    res = run_backtest(bars, Wild(), EngineConfig(mode="spot", cost_bps=0.0, holding_cost_annual=0.0))
    # long-only: a short target is clipped to 0, so equity never changes
    assert all(abs(eq - res.equity_curve[0]) < 1e-9 for eq in res.equity_curve)


def test_futures_short_profits_when_price_falls():
    class AlwaysShort:
        def on_bar(self, bar, macro):
            return -1.0

    bars = make_bars([100.0, 100.0, 90.0, 80.0])
    res = run_backtest(bars, AlwaysShort(), EngineConfig(mode="futures", cost_bps=0.0, holding_cost_annual=0.0))
    assert res.final_equity > res.equity_curve[0]


def test_macro_view_is_point_in_time():
    mv = MacroView([(date(2020, 2, 1), 0.02), (date(2020, 3, 1), 0.05)])
    assert mv.value_at(date(2020, 1, 15)) is None
    assert mv.value_at(date(2020, 2, 15)) == 0.02
    assert mv.value_at(date(2020, 3, 15)) == 0.05
    assert mv.prev_value() == 0.02
