"""Download public market datasets and normalize them to a common schema.

Normalized schema (CSV): ts,open,high,low,close,volume
- Close-only sources get open=high=low=close and volume=0.
- Macro series (CPI) use schema: ts,value

Sources (all public, fetched over HTTPS):
- BTC daily:        coinmetrics/data (PriceUSD)
- S&P500 daily:     vega/vega-datasets (sp500-2000.csv, OHLCV 2000-2020)
- WTI crude daily:  datasets/oil-prices
- Gold monthly:     datasets/gold-prices
- S&P500 monthly:   datasets/s-and-p-500 (Shiller; also provides CPI)
- VIX daily:        datasets/finance-vix

Run: python -m apps.worker.main fetch-data [--skip-download]
"""
from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/market")
OUT_DIR = Path("data/market/normalized")

SOURCES = {
    "btc_daily_raw.csv": "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
    "sp500_daily.csv": "https://raw.githubusercontent.com/vega/vega-datasets/main/data/sp500-2000.csv",
    "wti_daily.csv": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv",
    "gold_monthly.csv": "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
    "vix_daily.csv": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
    "shiller_sp500_cpi_monthly.csv": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
}


def download_all(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = raw_dir / name
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"[OK] downloaded {name} ({dest.stat().st_size} bytes)")


def _write_bars(path: Path, rows: list[tuple[str, float, float, float, float, float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow(r)
    return len(rows)


def _month_to_day(ym: str) -> str:
    # "1833-01" -> "1833-01-01"; already-dated strings pass through
    return ym if len(ym) == 10 else f"{ym}-01"


def normalize_all(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}

    # BTC daily: coinmetrics wide CSV, keep time + PriceUSD
    rows = []
    with (raw_dir / "btc_daily_raw.csv").open() as f:
        for rec in csv.DictReader(f):
            p = rec.get("PriceUSD", "")
            if p in ("", "NaN"):
                continue
            px = float(p)
            rows.append((rec["time"], px, px, px, px, 0.0))
    counts["BTC_1d"] = _write_bars(out_dir / "BTC_1d.csv", rows)

    # S&P500 daily OHLCV
    rows = []
    with (raw_dir / "sp500_daily.csv").open() as f:
        for rec in csv.DictReader(f):
            rows.append(
                (
                    rec["date"],
                    float(rec["open"]),
                    float(rec["high"]),
                    float(rec["low"]),
                    float(rec["close"]),
                    float(rec["volume"]),
                )
            )
    counts["SPX_1d"] = _write_bars(out_dir / "SPX_1d.csv", rows)

    # WTI daily close-only
    rows = []
    with (raw_dir / "wti_daily.csv").open() as f:
        for rec in csv.DictReader(f):
            if not rec["Price"]:
                continue
            px = float(rec["Price"])
            if px <= 0:  # April 2020 negative prints break log-returns; skip
                continue
            rows.append((rec["Date"], px, px, px, px, 0.0))
    counts["WTI_1d"] = _write_bars(out_dir / "WTI_1d.csv", rows)

    # Gold monthly close-only
    rows = []
    with (raw_dir / "gold_monthly.csv").open() as f:
        for rec in csv.DictReader(f):
            if not rec["Price"]:
                continue
            px = float(rec["Price"])
            rows.append((_month_to_day(rec["Date"]), px, px, px, px, 0.0))
    counts["GOLD_1mo"] = _write_bars(out_dir / "GOLD_1mo.csv", rows)

    # VIX daily OHLC (auxiliary regime data)
    rows = []
    with (raw_dir / "vix_daily.csv").open() as f:
        for rec in csv.DictReader(f):
            rows.append(
                (
                    rec["DATE"],
                    float(rec["OPEN"]),
                    float(rec["HIGH"]),
                    float(rec["LOW"]),
                    float(rec["CLOSE"]),
                    0.0,
                )
            )
    counts["VIX_1d"] = _write_bars(out_dir / "VIX_1d.csv", rows)

    # Shiller: S&P500 monthly close + CPI + long rate (zeros = not yet published)
    spx_rows = []
    cpi_rows = []
    rate_rows = []
    with (raw_dir / "shiller_sp500_cpi_monthly.csv").open() as f:
        for rec in csv.DictReader(f):
            ts = _month_to_day(rec["Date"])
            px = float(rec["SP500"] or 0)
            if px > 0:
                spx_rows.append((ts, px, px, px, px, 0.0))
            cpi = float(rec["Consumer Price Index"] or 0)
            if cpi > 0:
                cpi_rows.append((ts, cpi))
            rate = float(rec["Long Interest Rate"] or 0)
            if rate > 0:
                rate_rows.append((ts, rate))
    counts["SPX_1mo"] = _write_bars(out_dir / "SPX_1mo.csv", spx_rows)

    with (out_dir / "RATE_1mo.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "value"])  # US 10y yield, percent
        for r in rate_rows:
            w.writerow(r)
    counts["RATE_1mo"] = len(rate_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "CPI_1mo.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "value"])
        for r in cpi_rows:
            w.writerow(r)
    counts["CPI_1mo"] = len(cpi_rows)

    return counts


def fetch_market_data(skip_download: bool = False) -> dict[str, int]:
    if not skip_download:
        download_all()
    return normalize_all()


if __name__ == "__main__":
    print(fetch_market_data())
