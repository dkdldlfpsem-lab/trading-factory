from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from core.db import SessionLocal, init_db
from core.models import Feature, RawOHLCV, Signal, BacktestReport


def _payload_to_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return json.loads(x)
    return {}


def generate_signals_sma_ls(
    symbol: str,
    target_vol_annual: float = 0.20,
    max_leverage: float = 3.0,
    strength_norm: float = 0.02,  # diff/close 가 2%면 strength=1
) -> int:
    """
    SMA_fast vs SMA_slow로 Long/Short 시그널 생성.
    포지션 = sign * leverage * strength
    sign = +1 (fast>slow), -1 (fast<slow), 0 (같으면 0)
    leverage = clip(target_vol / realized_vol, 0..max_leverage)
    strength = clip(abs((fast-slow)/close)/strength_norm, 0..1)
    """
    init_db()

    with SessionLocal() as session:
        feats = (
            session.query(Feature)
            .filter(Feature.symbol == symbol)
            .order_by(Feature.ts.asc())
            .all()
        )

    if not feats:
        return 0

    df = pd.DataFrame(
        [{"ts": f.ts, "payload": _payload_to_dict(f.payload)} for f in feats]
    )
    df["SMA_fast"] = df["payload"].apply(lambda p: p.get("SMA_fast"))
    df["SMA_slow"] = df["payload"].apply(lambda p: p.get("SMA_slow"))
    df["realized_vol"] = df["payload"].apply(lambda p: p.get("realized_vol"))
    df["close"] = df["payload"].apply(lambda p: p.get("close"))  # 없으면 아래에서 raw로 채움

    # close가 payload에 없을 수 있으니 raw에서 붙인다
    with SessionLocal() as session:
        raws = (
            session.query(RawOHLCV.ts, RawOHLCV.close)
            .filter(RawOHLCV.symbol == symbol)
            .order_by(RawOHLCV.ts.asc())
            .all()
        )
    raw_df = pd.DataFrame(raws, columns=["ts", "close"])
    df = df.merge(raw_df, on="ts", how="left", suffixes=("", "_raw"))
    df["close"] = df["close"].fillna(df["close_raw"])
    df = df.drop(columns=["close_raw"])

    # sign
    df["sign"] = 0.0
    df.loc[df["SMA_fast"] > df["SMA_slow"], "sign"] = 1.0
    df.loc[df["SMA_fast"] < df["SMA_slow"], "sign"] = -1.0

    # strength
    diff = (df["SMA_fast"] - df["SMA_slow"]).abs()
    rel = diff / df["close"]
    df["strength"] = (rel / strength_norm).clip(lower=0.0, upper=1.0)

    # leverage (vol targeting)
    df["leverage"] = (target_vol_annual / df["realized_vol"]).clip(lower=0.0, upper=max_leverage)

    # position
    df["position"] = df["sign"] * df["strength"] * df["leverage"]

    inserted = 0
    with SessionLocal() as session:
        for _, r in df.iterrows():
            payload: Dict[str, float] = {
                "signal_type": "SMA_LS",
                "sign": float(r["sign"]) if pd.notna(r["sign"]) else 0.0,
                "strength": float(r["strength"]) if pd.notna(r["strength"]) else 0.0,
                "leverage": float(r["leverage"]) if pd.notna(r["leverage"]) else 0.0,
                "position": float(r["position"]) if pd.notna(r["position"]) else 0.0,
            }
            s = Signal(symbol=symbol, ts=r["ts"], payload=payload, version=1)
            session.merge(s)  # upsert
            inserted += 1
        session.commit()

    return inserted


def backtest_from_signals(
    symbol: str,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    report_dir: str | Path = "reports",
) -> dict:
    """
    - 수익률 = position(t-1) * return(t)
    - 비용 = turnover * (fee+slippage) bps
    turnover = abs(position - prev_position)
    """
    init_db()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine("sqlite:///./local.db")

    # raw
    raw = pd.read_sql(
        text("select ts, close from raw_ohlcv where symbol=:s order by ts asc"),
        engine,
        params={"s": symbol},
    )
    raw["ts"] = pd.to_datetime(raw["ts"])
    raw["ret"] = raw["close"].pct_change()

    # signals
    sig = pd.read_sql(
        text("select ts, payload from signals where symbol=:s order by ts asc"),
        engine,
        params={"s": symbol},
    )
    sig["ts"] = pd.to_datetime(sig["ts"])
    sig["payload"] = sig["payload"].apply(_payload_to_dict)
    sig["position"] = sig["payload"].apply(lambda p: float(p.get("position", 0.0)))

    df = raw.merge(sig[["ts", "position"]], on="ts", how="left").sort_values("ts")
    df["position"] = df["position"].fillna(0.0)

    # look-ahead 방지: 포지션은 한 칸 늦춰서 적용
    df["pos_lag"] = df["position"].shift(1).fillna(0.0)

    # turnover 비용
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    df["cost"] = df["turnover"] * cost_rate

    df["pnl"] = (df["pos_lag"] * df["ret"]).fillna(0.0) - df["cost"]
    df["equity"] = (1.0 + df["pnl"]).cumprod()

    total_return = float(df["equity"].iloc[-1] - 1.0)
    peak = df["equity"].cummax()
    dd = df["equity"] / peak - 1.0
    mdd = float(dd.min())

    win_rate = float((df["pnl"] > 0).mean())

    summary = {
        "symbol": symbol,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "total_return": total_return,
        "mdd": mdd,
        "win_rate": win_rate,
        "n_bars": int(len(df)),
        "created_at": datetime.now(timezone.utc)
.isoformat(),
    }

    # DB에 보고서 저장
    with SessionLocal() as session:
        session.add(BacktestReport(symbol=symbol, created_at=datetime.now(timezone.utc)
, payload=summary))
        session.commit()

    # 파일 저장
    out_path = report_dir / f"backtest_{symbol}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
