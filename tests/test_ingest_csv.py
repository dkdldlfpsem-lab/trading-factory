from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from core.db import init_db
from apps.worker.jobs.ingest_csv import ingest_csv


def _reset_local_db():
    # 로컬 DB를 테스트마다 깨끗하게 (간단 MVP)
    # Windows에서 파일 잠김 이슈 있으면 local.db 삭제 대신 테이블 비우기로 전환 가능
    init_db()
    engine = create_engine("sqlite:///./local.db")
    with engine.begin() as conn:
        conn.execute(text("delete from quality_events"))
        conn.execute(text("delete from raw_ohlcv"))


def test_ingest_ok():
    _reset_local_db()
    res = ingest_csv("data/sample_ohlcv.csv", symbol="SAMPLE")
    assert res.inserted > 0

    engine = create_engine("sqlite:///./local.db")
    with engine.connect() as conn:
        n = conn.execute(text("select count(*) from raw_ohlcv")).scalar_one()
    assert n > 0


def test_detect_duplicate_ts(tmp_path: Path):
    _reset_local_db()

    df = pd.read_csv("data/sample_ohlcv.csv")
    assert len(df) >= 2

    # 인위적 중복 ts 생성
    df.loc[1, "ts"] = df.loc[0, "ts"]
    p = tmp_path / "dup.csv"
    df.to_csv(p, index=False)

    res = ingest_csv(p, symbol="SAMPLE")
    assert res.qc_events >= 1

    engine = create_engine("sqlite:///./local.db")
    with engine.connect() as conn:
        n = conn.execute(
            text("select count(*) from quality_events where event_type='duplicate_ts'")
        ).scalar_one()
    assert n >= 1
