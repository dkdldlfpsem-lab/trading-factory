import json
from sqlalchemy import create_engine, text

from core.db import init_db
from apps.worker.jobs.ingest_csv import ingest_csv
from apps.worker.jobs.features import compute_features_for_symbol
from apps.worker.jobs.strategy_sma import generate_signals_sma_ls, backtest_from_signals


def _reset_db():
    init_db()
    engine = create_engine("sqlite:///./local.db")
    with engine.begin() as conn:
        conn.execute(text("delete from backtest_reports"))
        conn.execute(text("delete from signals"))
        conn.execute(text("delete from features"))
        conn.execute(text("delete from quality_events"))
        conn.execute(text("delete from raw_ohlcv"))


def test_signals_saved():
    _reset_db()

    ingest_csv("data/sample_ohlcv.csv", symbol="SAMPLE")
    compute_features_for_symbol("SAMPLE")

    n = generate_signals_sma_ls("SAMPLE")
    assert n > 0

    engine = create_engine("sqlite:///./local.db")
    with engine.connect() as conn:
        cnt = conn.execute(text("select count(*) from signals where symbol='SAMPLE'")).scalar_one()
    assert cnt > 0


def test_backtest_generates_metrics():
    _reset_db()

    ingest_csv("data/sample_ohlcv.csv", symbol="SAMPLE")
    compute_features_for_symbol("SAMPLE")
    generate_signals_sma_ls("SAMPLE")

    summary = backtest_from_signals("SAMPLE", fee_bps=1.0, slippage_bps=1.0)
    assert "total_return" in summary
    assert "mdd" in summary
