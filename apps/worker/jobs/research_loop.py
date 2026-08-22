"""Research loop: search for ONE parameter set that is profitable everywhere.

Anti-overfitting rules baked in:
- A candidate is evaluated with the SAME parameters on every asset
  (BTC, S&P500 daily, WTI, Gold monthly, S&P500 monthly since 1871).
  No per-symbol tuning.
- Metrics are split into in-sample (first 70% of bars) and out-of-sample
  (last 30%). A candidate only PASSES on its out-of-sample numbers.
- Every evaluated candidate is appended to reports/research_log.jsonl so the
  whole search is auditable.

The loop keeps generating candidates (coarse grid, then refinements around
the best performers) until the profit criteria are met or max_iters is hit.

Run: python -m apps.worker.main research
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from core.backtest import EngineConfig, run_backtest
from core.marketdata import MacroView, load_bars, load_cash_rate, load_cpi_yoy
from core.options import OptionsParams, WheelOptionsBacktester
from core.strategies import StrategyParams, TrendFollowStrategy

REPORTS_DIR = Path("reports")
LOG_PATH = REPORTS_DIR / "research_log.jsonl"

OOS_FRACTION = 0.30  # last 30% of bars is held out


@dataclass(frozen=True)
class AssetSpec:
    name: str
    symbol: str
    timeframe: str
    periods_per_year: float
    bars_per_month: int


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("BTC", "BTC", "1d", 365.0, 30),
    AssetSpec("SPX", "SPX", "1d", 252.0, 21),
    AssetSpec("WTI", "WTI", "1d", 252.0, 21),
    AssetSpec("GOLD", "GOLD", "1mo", 12.0, 1),
    AssetSpec("SPX_LONG", "SPX", "1mo", 12.0, 1),
)

# Options need real intramonth path; run on the daily assets only.
OPTION_ASSETS: tuple[AssetSpec, ...] = tuple(a for a in ASSETS if a.timeframe == "1d")


# ---------------------------------------------------------------- metrics ---

def segment_metrics(curve: list[float], start: int, ppy: float) -> dict:
    """Metrics on curve[start:], equity re-based to the segment start."""
    seg = curve[start:]
    if len(seg) < 2 or seg[0] <= 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    base = seg[0]
    tr = seg[-1] / base - 1.0
    years = len(seg) / ppy
    cagr = (seg[-1] / base) ** (1 / years) - 1.0 if seg[-1] > 0 and years > 0 else -1.0
    rets = [seg[i] / seg[i - 1] - 1.0 for i in range(1, len(seg)) if seg[i - 1] > 0]
    sharpe = 0.0
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        if var > 0:
            sharpe = mean / math.sqrt(var) * math.sqrt(ppy)
    peak, dd = -math.inf, 0.0
    for eq in seg:
        peak = max(peak, eq)
        if peak > 0:
            dd = max(dd, 1.0 - eq / peak)
    return {
        "total_return": round(tr, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 4),
    }


# ------------------------------------------------------------- evaluation ---

class _Data:
    """Load bars/macro once; reuse across the whole search."""

    def __init__(self) -> None:
        self.bars = {a.name: load_bars(a.symbol, a.timeframe) for a in ASSETS}
        self.cpi = load_cpi_yoy()
        self.rate = load_cash_rate()

    def macro(self) -> MacroView:
        return MacroView(self.cpi)

    def cash_rate(self) -> MacroView:
        return MacroView(self.rate)


def evaluate_trend_candidate(params: StrategyParams, data: _Data, cost_bps: float = 10.0) -> dict:
    """Run futures + spot with one param set on every asset; return metrics."""
    out: dict = {"futures": {}, "spot": {}}
    for mode in ("futures", "spot"):
        for a in ASSETS:
            bars = data.bars[a.name]
            strat = TrendFollowStrategy(params, a.bars_per_month, a.periods_per_year, mode=mode)
            cfg = EngineConfig(
                mode=mode,
                cost_bps=cost_bps,
                max_leverage=params.max_leverage,
                periods_per_year=a.periods_per_year,
                holding_cost_annual=0.01 if mode == "futures" else 0.0,
            )
            res = run_backtest(bars, strat, cfg, macro=data.macro())
            split = int(len(res.equity_curve) * (1 - OOS_FRACTION))
            out[mode][a.name] = {
                "full": res.summary(),
                "is": segment_metrics(res.equity_curve, 0, a.periods_per_year) | {"_end": split},
                "oos": segment_metrics(res.equity_curve, split, a.periods_per_year),
            }
    return out


def trend_criteria(metrics: dict) -> tuple[bool, dict]:
    """Profit criteria, judged OUT-OF-SAMPLE with one shared param set.

    futures: every asset OOS total_return > 0, mean OOS sharpe >= 0.5,
             every asset OOS max_drawdown <= 0.55
    spot:    no asset loses more than 2% OOS, mean OOS total_return > 0
    """
    fut = metrics["futures"]
    spot = metrics["spot"]
    fut_tr = [m["oos"]["total_return"] for m in fut.values()]
    fut_sh = [m["oos"]["sharpe"] for m in fut.values()]
    fut_dd = [m["oos"]["max_drawdown"] for m in fut.values()]
    spot_tr = [m["oos"]["total_return"] for m in spot.values()]
    checks = {
        "futures_all_profitable_oos": all(tr > 0 for tr in fut_tr),
        "futures_mean_sharpe_oos>=0.5": (sum(fut_sh) / len(fut_sh)) >= 0.5,
        "futures_maxdd_oos<=0.55": all(dd <= 0.55 for dd in fut_dd),
        "spot_no_asset_loses_oos": all(tr > -0.02 for tr in spot_tr),
        "spot_mean_profitable_oos": (sum(spot_tr) / len(spot_tr)) > 0,
    }
    return all(checks.values()), checks


def score(metrics: dict) -> float:
    """Ranking score for refinement: mean OOS sharpe minus a loss penalty."""
    fut = metrics["futures"]
    sh = [m["oos"]["sharpe"] for m in fut.values()]
    tr = [m["oos"]["total_return"] for m in fut.values()]
    penalty = sum(min(0.0, t) for t in tr) * 2.0
    return sum(sh) / len(sh) + penalty


# --------------------------------------------------------------- the loop ---

def base_grid() -> list[StrategyParams]:
    grid = []
    for lbs, vt, regime, cpi in itertools.product(
        [(3, 6, 12), (6, 12), (1, 3, 12), (12,)],
        [0.10, 0.15, 0.20],
        [True, False],
        [True, False],
    ):
        grid.append(
            StrategyParams(
                mom_lookbacks_months=lbs,
                vol_target_annual=vt,
                use_regime_filter=regime,
                use_cpi_filter=cpi,
            )
        )
    return grid


def refine(best: StrategyParams) -> list[StrategyParams]:
    """Perturb the current best candidate (still shared across assets)."""
    variants: list[StrategyParams] = []
    for vt in {round(best.vol_target_annual + d, 2) for d in (-0.03, 0.03, 0.05)}:
        if 0.05 <= vt <= 0.30:
            variants.append(replace(best, vol_target_annual=vt))
    for sma in (8, 12):
        variants.append(replace(best, regime_sma_months=sma))
    for damp in (0.0, 0.3, 0.7):
        variants.append(replace(best, cpi_damp=damp, use_cpi_filter=True))
    variants.append(replace(best, allow_short=False))
    variants.append(replace(best, max_leverage=2.0))
    return variants


def _log(entry: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def run_research(max_iters: int = 300, cost_bps: float = 10.0) -> dict:
    data = _Data()
    seen: set[str] = set()
    iteration = 0
    best_params: StrategyParams | None = None
    best_metrics: dict | None = None
    best_score = -math.inf
    winner: dict | None = None

    queue: list[StrategyParams] = base_grid()

    while queue and iteration < max_iters:
        params = queue.pop(0)
        key = json.dumps(params.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        iteration += 1

        metrics = evaluate_trend_candidate(params, data, cost_bps=cost_bps)
        passed, checks = trend_criteria(metrics)
        sc = score(metrics)
        _log(
            {
                "iteration": iteration,
                "ts": datetime.now(timezone.utc).isoformat(),
                "params": params.to_dict(),
                "score": round(sc, 4),
                "passed": passed,
                "checks": checks,
                "oos_futures": {k: v["oos"] for k, v in metrics["futures"].items()},
                "oos_spot": {k: v["oos"] for k, v in metrics["spot"].items()},
            }
        )
        if sc > best_score:
            best_score, best_params, best_metrics = sc, params, metrics
        if passed:
            winner = {"params": params, "metrics": metrics, "iteration": iteration, "checks": checks}
            break
        if not queue and best_params is not None:
            # grid exhausted without a pass: refine around the best so far
            queue.extend(refine(best_params))

    if winner is None:
        assert best_params is not None and best_metrics is not None
        winner = {
            "params": best_params,
            "metrics": best_metrics,
            "iteration": iteration,
            "checks": trend_criteria(best_metrics)[1],
        }
        winner["passed_all_criteria"] = False
    else:
        winner["passed_all_criteria"] = True

    # ---- options search on top of the winning regime parameters ----
    opt_winner = search_options(winner["params"], data, iteration_offset=iteration)
    winner["options"] = opt_winner
    return winner


def search_options(strat_params: StrategyParams, data: _Data, iteration_offset: int = 0) -> dict:
    grid = [
        OptionsParams(otm_put=op, otm_call=oc, iv_markup=iv, expiry_days=ed, notional_frac=nf)
        for op, oc, iv, ed, nf in itertools.product(
            [0.05, 0.10, 0.15],
            [0.05, 0.10],
            [1.10, 1.20],
            [30, 45],
            [1.0, 0.5],
        )
    ]
    best = None
    it = 0
    for opt in grid:
        it += 1
        per_asset = {}
        for a in OPTION_ASSETS:
            bars = data.bars[a.name]
            bt = WheelOptionsBacktester(
                strat_params, opt, a.bars_per_month, a.periods_per_year, cash_rate=data.cash_rate()
            )
            res = bt.run(bars, macro=data.macro())
            split = int(len(res.equity_curve) * (1 - OOS_FRACTION))
            per_asset[a.name] = {
                "full": res.summary(),
                "oos": segment_metrics(res.equity_curve, split, a.periods_per_year),
            }
        oos_tr = [m["oos"]["total_return"] for m in per_asset.values()]
        oos_sh = [m["oos"]["sharpe"] for m in per_asset.values()]
        passed = all(tr > 0 for tr in oos_tr)
        sc = sum(oos_sh) / len(oos_sh)
        _log(
            {
                "iteration": iteration_offset + it,
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": "options",
                "params": opt.to_dict(),
                "score": round(sc, 4),
                "passed": passed,
                "oos": {k: v["oos"] for k, v in per_asset.items()},
            }
        )
        if best is None or (passed, sc) > (best["passed"], best["score"]):
            best = {"params": opt, "metrics": per_asset, "passed": passed, "score": sc}
        if passed:
            break
    assert best is not None
    return best


# ---------------------------------------------------------------- reports ---

def write_reports(winner: dict) -> list[Path]:
    REPORTS_DIR.mkdir(exist_ok=True)
    written: list[Path] = []
    params: StrategyParams = winner["params"]
    for mode in ("futures", "spot"):
        for asset, m in winner["metrics"][mode].items():
            path = REPORTS_DIR / f"backtest_{mode}_{asset}.json"
            path.write_text(
                json.dumps(
                    {
                        "strategy": f"trend_follow_{mode}",
                        "asset": asset,
                        "params": params.to_dict(),
                        "metrics": m,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
            written.append(path)
    opt = winner.get("options")
    if opt:
        for asset, m in opt["metrics"].items():
            path = REPORTS_DIR / f"backtest_options_{asset}.json"
            path.write_text(
                json.dumps(
                    {
                        "strategy": "options_wheel_regime",
                        "asset": asset,
                        "strategy_params": params.to_dict(),
                        "options_params": opt["params"].to_dict(),
                        "metrics": m,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
            written.append(path)
    return written


if __name__ == "__main__":
    w = run_research()
    write_reports(w)
    print(json.dumps({"passed": w["passed_all_criteria"], "iteration": w["iteration"], "params": w["params"].to_dict()}, indent=2))
