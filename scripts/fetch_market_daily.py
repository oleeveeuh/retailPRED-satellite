#!/usr/bin/env python3
"""Fetch raw daily market bars for the H9 market rung (H9_FEATURE_REVISIT.md §5).

Predeclared tickers (v1's exact list): AAPL, AMZN, WMT, COST, SPY, QQQ,
XLY, XRT. Source: the Yahoo chart API via plain requests — **as-traded
close and volume only** (no adjusted fields, no auto-adjust), because
market quotes are facts: unlike adjusted closes they are never
retroactively revised, which is what makes the vintage discipline of
H9 §3 well-defined. Output: data/market_daily_closes.csv
(symbol, date, close, volume, ingested_at, source); the SHA-256 and
per-symbol span are recorded in results/h9_market_ingestion.json.

If the API cannot complete, the market rung is reported *implemented, not
run* per H9 §5 — no synthetic substitution is ever permitted.

Protocol predeclaration: ticker list, fields, span, and output schema were
frozen in H9_FEATURE_REVISIT.md §5 before any fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, git_revision

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Predeclared tickers — v1-undergrad backend/ml/external_data_loader.py.
H9_TICKERS = ("AAPL", "AMZN", "WMT", "COST", "SPY", "QQQ", "XLY", "XRT")
SPAN_START = date(2005, 1, 1)
OUTPUT_PATH = ROOT / "data" / "market_daily_closes.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (research script; retailPRED-satellite)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def fetch_symbol(session: requests.Session, symbol: str) -> pd.DataFrame:
    params = {
        "period1": int(datetime(SPAN_START.year, SPAN_START.month, SPAN_START.day, tzinfo=timezone.utc).timestamp()),
        "period2": int(datetime.now(timezone.utc).timestamp()),
        "interval": "1d",
        "events": "history",
    }
    response = session.get(CHART_URL.format(symbol=symbol), params=params,
                           headers=HEADERS, timeout=60)
    if response.status_code != 200:
        raise SystemExit(
            f"chart request for {symbol} failed with HTTP {response.status_code}; "
            "per H9 §5 the market rung is then reported implemented-not-run — "
            "no synthetic substitution"
        )
    result = response.json().get("chart", {}).get("result") or []
    if not result:
        raise SystemExit(f"chart response for {symbol} carried no result payload")
    timestamps = result[0].get("timestamp") or []
    quote = (result[0].get("indicators") or {}).get("quote") or [{}]
    closes = quote[0].get("close") or []
    volumes = quote[0].get("volume") or []
    rows = []
    for position, stamp in enumerate(timestamps):
        close = closes[position] if position < len(closes) else None
        volume = volumes[position] if position < len(volumes) else None
        if close is None or volume is None:
            continue  # halted/missing sessions are absent facts, never imputed
        rows.append(
            {
                "symbol": symbol,
                "date": datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat(),
                "close": float(close),
                "volume": float(volume),
                "ingested_at": None,
                "source": "yahoo_chart_raw",
            }
        )
    if not rows:
        raise SystemExit(f"symbol {symbol} produced zero usable sessions")
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    session = requests.Session()
    frames = []
    provenance_symbols: dict[str, dict] = {}
    for symbol in H9_TICKERS:
        frame = fetch_symbol(session, symbol)
        frames.append(frame)
        dates = frame["date"].sort_values()
        provenance_symbols[symbol] = {
            "sessions": int(len(frame)),
            "first": dates.iloc[0],
            "last": dates.iloc[-1],
        }
        print(f"  {symbol}: {len(frame):,} sessions ({dates.iloc[0]} → {dates.iloc[-1]})", flush=True)
        time.sleep(1.0)

    ingested_at = datetime.now(timezone.utc)
    result = pd.concat(frames, ignore_index=True)
    result["ingested_at"] = ingested_at
    result = result.sort_values(["symbol", "date"], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    payload = {
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": digest,
        "row_count": int(len(result)),
        "symbols": provenance_symbols,
        "field_discipline": "as-traded close and volume only; no adjusted fields",
        "satellite_code_revision": git_revision(),
        "generated_at": ingested_at.isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "h9_market_ingestion.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    print(f"Wrote {len(result):,} daily bars to {args.output.relative_to(ROOT)}")
    print(f"SHA-256 {digest} recorded in results/h9_market_ingestion.json")


if __name__ == "__main__":
    main()
