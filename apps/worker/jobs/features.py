from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from core.db import SessionLocal, init_db
from core.models import RawOHLCV, Feature


def compute_features_for_symbol(symbol: str) -> int:
    """
    raw_ohlcv -> features 테이블 저장
    look-ahead 금지: 모든 계산은 과거 데이터만 사용
    """
    init_db()

    with SessionLocal() as session:
        rows = (
            session.query(RawOHLCV)
            .filter(RawOHLCV.symbol == symbol)
            .order_by(RawOHLCV.ts.asc())
            .all()
        )

    if not rows:
        return 0

    df = pd.DataFrame(
        [
            {
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )

    df = df.sort_values("ts").reset_index(drop=True)

    # -------------------
    # Feature 계산 (look-ahead 없음)
    # -------------------

    df["returns"] = df["close"].pct_change()

    # realized volatility (rolling std)
    df["realized_vol"] = (
        df["returns"]
        .rolling(window=5, min_periods=5)
        .std()
        * math.sqrt(252)
    )

    # SMA
    df["SMA_fast"] = df["close"].rolling(5, min_periods=5).mean()
    df["SMA_slow"] = df["close"].rolling(20, min_periods=20).mean()

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["true_range"].rolling(14, min_periods=14).mean()

    # volume z-score
    vol_mean = df["volume"].rolling(20, min_periods=20).mean()
    vol_std = df["volume"].rolling(20, min_periods=20).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std

    # drawdown
    rolling_max = df["close"].cummax()
    df["drawdown"] = df["close"] / rolling_max - 1.0

    inserted = 0

    with SessionLocal() as session:
        for _, row in df.iterrows():
            payload: Dict[str, float] = {
                "returns": row["returns"],
                "realized_vol": row["realized_vol"],
                "ATR": row["ATR"],
                "SMA_fast": row["SMA_fast"],
                "SMA_slow": row["SMA_slow"],
                "volume_z": row["volume_z"],
                "drawdown": row["drawdown"],
            }

            f = Feature(
                symbol=symbol,
                ts=row["ts"],
                payload=payload,
                version=1,
            )

            session.merge(f)  # upsert
            inserted += 1

        session.commit()

    return inserted
