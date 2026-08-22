import json
import pandas as pd
from sqlalchemy import create_engine, text

from core.db import init_db
from apps.worker.jobs.ingest_csv import ingest_csv
from apps.worker.jobs.features import compute_features_for_symbol


def _reset_db():
    init_db()
    engine = create_engine("sqlite:///./local.db")
    with engine.begin() as conn:
        conn.execute(text("delete from features"))
        conn.execute(text("delete from quality_events"))
        conn.execute(text("delete from raw_ohlcv"))


def test_feature_generation():
    _reset_db()

    # raw ingest
    ingest_csv("data/sample_ohlcv.csv", symbol="SAMPLE")

    inserted = compute_features_for_symbol("SAMPLE")
    assert inserted > 0

    engine = create_engine("sqlite:///./local.db")
    with engine.connect() as conn:
        n = conn.execute(text("select count(*) from features")).scalar_one()
    assert n > 0


def test_no_lookahead_bias():
    """
    미래 데이터 참조하지 않는지 간단 검증:
    마지막 row의 returns는 (마지막 close / 직전 close - 1) 이어야 한다.
    """
    _reset_db()

    ingest_csv("data/sample_ohlcv.csv", symbol="SAMPLE")
    compute_features_for_symbol("SAMPLE")

    engine = create_engine("sqlite:///./local.db")

    with engine.connect() as conn:
        rows = conn.execute(
            text("select ts, payload from features where symbol='SAMPLE' order by ts asc")
        ).fetchall()

    assert len(rows) >= 2

    # payload가 SQLite에서 TEXT로 나올 수 있으므로 json.loads 처리
    last_payload = rows[-1][1]
    if isinstance(last_payload, str):
        last_payload = json.loads(last_payload)

    db_ret = last_payload["returns"]

    # raw 기준 계산
    df = pd.read_csv("data/sample_ohlcv.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    calc_ret = df["returns"].iloc[-1]

    if pd.isna(calc_ret):
        assert db_ret is None or pd.isna(db_ret)
    else:
        assert abs(db_ret - calc_ret) < 1e-10
