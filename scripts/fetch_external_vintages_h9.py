#!/usr/bin/env python3
"""Fetch vintage-safe macro/spending vintages from ALFRED (H9_FEATURE_REVISIT.md §5).

Predeclared series (monthly frequency only, ALFRED-archived) — the union of
v1's economic-indicator axis, the H7 external five, and v1's
consumer-spending axis: CPIAUCSL, UNRATE, UMCSENT, FEDFUNDS, DSPI, MSL,
INDPRO, PCE, PCEDG. Identical machinery and sampling rule to the H7
ingestion (one full history per archived vintage date; the LAST archived
vintage per calendar month is kept). Output: data/alfred_external_vintages_h9.csv
in the canonical panel schema; the H7 file is never touched.

Provenance: records the output SHA-256, per-series vintage counts, the
vintage range, and package versions in results/h9_external_ingestion.json.
Missing credentials fail loudly; no synthetic data is ever substituted.
A listed series ALFRED does not archive aborts ingestion (reported, never
substituted) per H9 §5.

Protocol predeclaration: series list, sampling rule, and range were frozen
in H9_FEATURE_REVISIT.md §5 before any fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, git_revision
from fetch_external_vintages import OBSERVATION_START, VintageClient
import time as _time


class RetryingVintageClient(VintageClient):
    """The H7 client plus bounded retry/backoff for transient connection
    drops (a ~1,500-request run must not die on one reset). Retry covers
    transport errors and 5xx only — a 4xx is a real API error and fails
    loudly, exactly as before."""

    def _get(self, url: str, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return super()._get(url, params)
            except SystemExit:
                raise
            except Exception as error:  # transport-level failure
                last_error = error
                _time.sleep(2.0 * (attempt + 1))
        raise SystemExit(f"FRED request failed after retries: {last_error}")

# our context id -> FRED series id (predeclared H9_FEATURE_REVISIT.md §3/§5)
H9_EXTERNAL_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",
    "dspi": "DSPI",
    "umsent": "UMCSENT",
    "unrate": "UNRATE",
    "fedfunds": "FEDFUNDS",
    "m2": "MSL",
    "indpro": "INDPRO",
    "pce": "PCE",
}
# Dropped per the predeclared §5 rule (execution record 2026-11-11): PCEDG
# is not ALFRED-archived — the vintage-dates endpoint returns only the
# current date and observations fail with the ALFRED-absence error. Dropped
# and reported; never substituted. Recorded in the ingestion provenance.
DROPPED_SERIES: dict[str, str] = {
    "pcedg": "PCEDG — not ALFRED-archived (no vintage history; observations "
             "reject realtime_start queries)",
}
OUTPUT_PATH = ROOT / "data" / "alfred_external_vintages_h9.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1995,
                        help="first vintage year to fetch (predeclared: 1995)")
    parser.add_argument("--end-year", type=int, required=True,
                        help="last vintage year to fetch, e.g. 2026")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_year < 1992:
        raise SystemExit("--start-year cannot precede the panel's 1992 observation start")
    client = RetryingVintageClient(os.getenv("FRED_API_KEY", ""))
    lower = date(args.start_year, 1, 1)
    upper = date(args.end_year, 12, 31)

    frames: list[pd.DataFrame] = []
    provenance_series: dict[str, dict] = {}
    total_requests = 0
    completed = 0
    for context_id, fred_id in sorted(H9_EXTERNAL_SERIES.items()):
        archived = [v for v in client.fetch_vintage_dates(fred_id) if lower <= v <= upper]
        if not archived:
            raise SystemExit(
                f"ALFRED does not archive {context_id} ({fred_id}) in range; "
                "per H9 §5 it must be dropped-and-reported, never substituted — "
                "rerun with the series removed via a predeclaration amendment "
                "committed before any run"
            )
        by_month: dict[tuple[int, int], date] = {}
        for vintage in archived:
            by_month[(vintage.year, vintage.month)] = vintage
        selected = sorted(by_month.values())
        total_requests += len(selected)
        provenance_series[context_id] = {
            "fred_id": fred_id,
            "vintage_count": len(selected),
            "vintage_first": selected[0].isoformat(),
            "vintage_last": selected[-1].isoformat(),
        }
        print(f"  {context_id} ({fred_id}): {len(selected)} monthly vintages", flush=True)
        for vintage_date in selected:
            payload = client.fetch_monthly_vintage(fred_id, vintage_date)
            rows = [
                {
                    "series_id": f"external_{context_id}",
                    "observation_time": pd.Timestamp(observation["date"]),
                    "value": float(observation["value"]),
                    "release_time": pd.NaT,
                    "vintage_time": pd.Timestamp(vintage_date).tz_localize("UTC"),
                    "ingested_at": None,
                    "source": "fred_alfred",
                }
                for observation in payload["observations"]
                if str(observation.get("value", ".")) not in {".", "", "NaN"}
            ]
            frames.append(pd.DataFrame.from_records(rows))
            completed += 1
            if completed % 100 == 0:
                print(f"  {completed}/{total_requests} vintage fetches complete", flush=True)
            time.sleep(0.4)  # FRED: no more than 120 requests per minute.

    ingested_at = datetime.now(timezone.utc)
    result = pd.concat(frames, ignore_index=True)
    result["ingested_at"] = ingested_at
    result = result.sort_values(
        ["series_id", "vintage_time", "observation_time"], ignore_index=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    payload = {
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": digest,
        "row_count": int(len(result)),
        "series": provenance_series,
        "dropped_series": DROPPED_SERIES,
        "total_vintage_fetches": total_requests,
        "sampling_rule": "last archived vintage per calendar month",
        "h7_file_untouched": True,
        "satellite_code_revision": git_revision(),
        "generated_at": ingested_at.isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "h9_external_ingestion.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    print(f"Wrote {len(result):,} versioned observations to {args.output.relative_to(ROOT)}")
    print(f"SHA-256 {digest} recorded in results/h9_external_ingestion.json")


if __name__ == "__main__":
    main()
