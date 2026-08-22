from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from core.db import SessionLocal, init_db
from core.models import RawOHLCV, QualityEvent


@dataclass
class IngestResult:
    attempted: int   # 이번에 넣으려고 시도한 행 수
    inserted: int    # 실제로 DB에 새로 들어간 행 수
    skipped: int     # 중복이라 무시된 행 수 (attempted - inserted)
    qc_events: int   # QC 이벤트 개수


def _utcnow_naive() -> datetime:
    """
    DeprecationWarning(datetime.utcnow) 제거용.
    UTC 기준으로 now(timezone.utc)를 쓰되, DB에 저장할 때 timezone-aware가 꼬일 수 있어
    naive(datetime)로 통일.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_ts(x: Any) -> datetime:
    # pandas Timestamp / str 모두 처리
    return pd.to_datetime(x, utc=False).to_pydatetime()


def run_quality_checks(df: pd.DataFrame, symbol: str) -> list[dict]:
    """
    df columns: ts, open, high, low, close, volume (ts is datetime64)
    """
    events: list[dict] = []

    # missing ts
    if df["ts"].isna().any():
        events.append(
            dict(
                symbol=symbol,
                ts=_utcnow_naive(),
                event_type="missing_ts",
                severity=3,
                payload={"count": int(df["ts"].isna().sum())},
            )
        )

    # duplicate ts
    dup_mask = df["ts"].duplicated(keep=False)
    if dup_mask.any():
        dup_ts = pd.to_datetime(df.loc[dup_mask, "ts"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        events.append(
            dict(
                symbol=symbol,
                ts=_parse_ts(df.loc[dup_mask, "ts"].iloc[0]),
                event_type="duplicate_ts",
                severity=3,
                payload={"count": int(dup_mask.sum()), "timestamps": dup_ts[:50]},
            )
        )

    # gap detection
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    if len(df_sorted) >= 3:
        dt = df_sorted["ts"].diff().dropna()
        base = dt.mode().iloc[0] if not dt.mode().empty else dt.median()
        gap_mask = dt > (base * 1.5)
        if gap_mask.any():
            idxs = gap_mask[gap_mask].index.tolist()[:20]
            for i in idxs:
                from_ts = df_sorted.loc[i - 1, "ts"]
                to_ts = df_sorted.loc[i, "ts"]
                gap_sec = int((to_ts - from_ts).total_seconds())
                events.append(
                    dict(
                        symbol=symbol,
                        ts=_parse_ts(to_ts),
                        event_type="gap",
                        severity=2,
                        payload={
                            "gap_seconds": gap_sec,
                            "from_ts": str(from_ts),
                            "to_ts": str(to_ts),
                        },
                    )
                )

    # spike detection: abs(close pct change) > 10%
    if len(df_sorted) >= 2:
        ret = df_sorted["close"].pct_change()
        spike_mask = ret.abs() > 0.10
        if spike_mask.any():
            idxs = spike_mask[spike_mask].index.tolist()[:20]
            for i in idxs:
                ts = df_sorted.loc[i, "ts"]
                prev_close = float(df_sorted.loc[i - 1, "close"])
                close = float(df_sorted.loc[i, "close"])
                events.append(
                    dict(
                        symbol=symbol,
                        ts=_parse_ts(ts),
                        event_type="spike",
                        severity=2,
                        payload={"ret": float(ret.loc[i]), "prev_close": prev_close, "close": close},
                    )
                )

    return events


def ingest_csv(csv_path: str | Path, symbol: str = "SAMPLE", source: str = "sample_csv") -> IngestResult:
    """
    Read CSV and insert into raw_ohlcv, run QC and store into quality_events.
    Assumes db is sqlite local.db by default.
    """
    init_db()

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # expected columns: symbol(optional), ts, open, high, low, close, volume
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "source" not in df.columns:
        df["source"] = source

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # QC는 "원본 df"로 돌리고
    qc_events = run_quality_checks(df[df["symbol"] == symbol], symbol=symbol)

    # raw insert는 (symbol, ts) 중복 제거 후 진행
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="first").reset_index(drop=True)

    attempted = 0
    inserted = 0
    skipped = 0

    with SessionLocal() as session:
        # ✅ raw_ohlcv UPSERT(충돌 시 무시)
        values: list[dict] = []
        for _, r in df.iterrows():
            values.append(
                dict(
                    symbol=str(r["symbol"]),
                    ts=_parse_ts(r["ts"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                    source=str(r["source"]),
                )
            )

        attempted = len(values)

        if values:
            stmt = sqlite_insert(RawOHLCV).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "ts"])
            result = session.execute(stmt)

            inserted = int(result.rowcount or 0)
            skipped = attempted - inserted

        # ✅ quality_events도 UPSERT(충돌 시 무시)
        # 전제: QualityEvent에 UNIQUE(symbol, ts, event_type) 제약이 있어야 함
        if qc_events:
            q_stmt = sqlite_insert(QualityEvent).values(qc_events)
            q_stmt = q_stmt.on_conflict_do_nothing(index_elements=["symbol", "ts", "event_type"])
            session.execute(q_stmt)

        session.commit()

    return IngestResult(
        attempted=attempted,
        inserted=inserted,
        skipped=skipped,
        qc_events=len(qc_events),
    )
