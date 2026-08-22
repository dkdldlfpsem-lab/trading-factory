from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, JSON, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawOHLCV(Base):
    __tablename__ = "raw_ohlcv"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    source: Mapped[str] = mapped_column(String)


class QualityEvent(Base):
    __tablename__ = "quality_events"

    # ✅ 중복 방지: 같은 (symbol, ts, event_type)는 한 번만 저장
    __table_args__ = (
        UniqueConstraint("symbol", "ts", "event_type", name="uq_quality_event_symbol_ts_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    severity: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)


class Feature(Base):
    __tablename__ = "features"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    payload: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)


class Signal(Base):
    __tablename__ = "signals"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    payload: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)


class BacktestReport(Base):
    __tablename__ = "backtest_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
