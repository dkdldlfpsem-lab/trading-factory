import subprocess
import sys

from sqlalchemy import create_engine, text

from core.db import init_db


def _reset_db():
    init_db()
    engine = create_engine("sqlite:///./local.db")
    with engine.begin() as conn:
        # 파이프라인 결과물 테이블 비우기
        conn.execute(text("delete from backtest_reports"))
        conn.execute(text("delete from signals"))
        conn.execute(text("delete from features"))
        conn.execute(text("delete from quality_events"))
        conn.execute(text("delete from raw_ohlcv"))


def test_runner_all_creates_signals():
    _reset_db()

    # 윈도우에서도 확실히 돌도록 sys.executable 사용
    cmd = [sys.executable, "-m", "apps.worker.main", "all", "--csv", "data/sample_ohlcv.csv", "--symbol", "SAMPLE"]
    r = subprocess.run(cmd, capture_output=True, text=True)

    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    engine = create_engine("sqlite:///./local.db")
    with engine.connect() as conn:
        n = conn.execute(text("select count(*) from signals where symbol='SAMPLE'")).scalar_one()
    assert n > 0
