from __future__ import annotations

import argparse
import sys
import traceback

from core.db import init_db
from apps.worker.jobs.ingest_csv import ingest_csv
from apps.worker.jobs.features import compute_features_for_symbol
from apps.worker.jobs.strategy_sma import generate_signals_sma_ls


def _fail(msg: str, exc: Exception | None = None) -> int:
    print(f"[ERROR] {msg}", file=sys.stderr)
    if exc is not None:
        traceback.print_exc()
    return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    init_db()
    res = ingest_csv(args.csv, symbol=args.symbol)
    print(f"[OK] ingest: inserted={res.inserted} qc_events={res.qc_events}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    init_db()
    n = compute_features_for_symbol(args.symbol)
    print(f"[OK] features: inserted={n}")
    return 0


def cmd_strategy(args: argparse.Namespace) -> int:
    init_db()
    n = generate_signals_sma_ls(args.symbol)
    print(f"[OK] strategy: signals_upserted={n}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    # ingest -> features -> strategy
    r = cmd_ingest(args)
    if r != 0:
        return r
    r = cmd_features(args)
    if r != 0:
        return r
    r = cmd_strategy(args)
    if r != 0:
        return r
    print("[OK] all: pipeline completed")
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from apps.worker.jobs.fetch_market_data import fetch_market_data

    counts = fetch_market_data(skip_download=args.skip_download)
    print(f"[OK] fetch-data: {counts}")
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    import json

    from apps.worker.jobs.research_loop import run_research, write_reports

    winner = run_research(max_iters=args.max_iters, cost_bps=args.cost_bps)
    written = write_reports(winner)
    print(
        f"[OK] research: passed={winner['passed_all_criteria']} "
        f"iterations={winner['iteration']} reports={len(written)}"
    )
    print(json.dumps(winner["params"].to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apps.worker.main", description="Worker pipeline runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ing = sub.add_parser("ingest", help="Ingest CSV into raw_ohlcv with QC")
    p_ing.add_argument("--csv", default="data/sample_ohlcv.csv", help="CSV path")
    p_ing.add_argument("--symbol", default="SAMPLE", help="symbol")
    p_ing.set_defaults(fn=cmd_ingest)

    # features
    p_feat = sub.add_parser("features", help="Compute features from raw_ohlcv")
    p_feat.add_argument("--symbol", default="SAMPLE", help="symbol")
    p_feat.set_defaults(fn=cmd_features)

    # strategy
    p_strat = sub.add_parser("strategy", help="Generate strategy signals from features")
    p_strat.add_argument("--symbol", default="SAMPLE", help="symbol")
    p_strat.set_defaults(fn=cmd_strategy)

    # all
    p_all = sub.add_parser("all", help="Run ingest -> features -> strategy")
    p_all.add_argument("--csv", default="data/sample_ohlcv.csv", help="CSV path")
    p_all.add_argument("--symbol", default="SAMPLE", help="symbol")
    p_all.set_defaults(fn=cmd_all)

    # fetch-data
    p_fetch = sub.add_parser("fetch-data", help="Download + normalize public market datasets")
    p_fetch.add_argument("--skip-download", action="store_true", help="Only re-normalize existing raw files")
    p_fetch.set_defaults(fn=cmd_fetch_data)

    # research
    p_res = sub.add_parser("research", help="Run the strategy research loop until profit criteria pass")
    p_res.add_argument("--max-iters", type=int, default=300)
    p_res.add_argument("--cost-bps", type=float, default=10.0)
    p_res.set_defaults(fn=cmd_research)

    return p


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return int(args.fn(args))
    except Exception as e:
        return _fail("runner crashed", e)


if __name__ == "__main__":
    raise SystemExit(main())
