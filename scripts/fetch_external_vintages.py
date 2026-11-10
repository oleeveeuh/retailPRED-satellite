#!/usr/bin/env python3
"""Fetch vintage-safe external macro vintages from ALFRED (H7_FEATURE_LADDER.md §5).

Predeclared series (monthly frequency only, ALFRED-archived): CPIAUCSL,
UNRATE, UMCSENT, FEDFUNDS, DSPI. One full history per archived vintage
date; the LAST archived vintage per calendar month is kept (the MRTS
panel's sampling rule, so origin as-of selection sees each month's final
published state). Output follows the canonical panel schema.

Provenance: records the output SHA-256, per-series vintage counts, the
vintage range, and package versions in results/h7_external_ingestion.json.
Missing credentials fail loudly; no synthetic data is ever substituted.

Protocol predeclaration: this script's series list, sampling rule, and
range were frozen in H7_FEATURE_LADDER.md §5 before any fetch; changing
them after external results are seen is a goalpost move.
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
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, git_revision

from vintageforensics.feature_groups import EXTERNAL_SERIES

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
OUTPUT_PATH = ROOT / "data" / "alfred_external_vintages.csv"
OBSERVATION_START = date(1992, 1, 1)  # mirror the MRTS panel's observation start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1995,
                        help="first vintage year to fetch (predeclared: 1995)")
    parser.add_argument("--end-year", type=int, required=True,
                        help="last vintage year to fetch, e.g. 2026")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


class VintageClient:
    """Minimal port of retailPRED forecasting.data.fred.FREDClient (subset)."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise SystemExit(
                "FRED_API_KEY is required in the environment for ALFRED ingestion; "
                "no synthetic substitution is permitted."
            )
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, url: str, params: dict) -> dict:
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code != 200:
            # FRED error bodies name the cause and never echo the api_key.
            raise SystemExit(
                f"FRED request failed with HTTP {response.status_code}: "
                f"{response.text.strip()[:200] or '(no body)'}"
            )
        return response.json()

    def fetch_vintage_dates(self, fred_id: str) -> list[date]:
        payload = self._get(FRED_VINTAGE_DATES_URL, {
            "series_id": fred_id,
            "api_key": self.api_key,
            "file_type": "json",
        })
        return [date.fromisoformat(item["date"]) for item in payload.get("vintage_dates", [])]

    def fetch_monthly_vintage(self, fred_id: str, vintage_date: date) -> dict:
        return self._get(FRED_OBSERVATIONS_URL, {
            "series_id": fred_id,
            "api_key": self.api_key,
            "file_type": "json",
            "realtime_start": vintage_date.isoformat(),
            "realtime_end": vintage_date.isoformat(),
            "frequency": "m",
            "aggregation_method": "avg",
            "units": "lin",
            "output_type": 1,
            "observation_start": OBSERVATION_START.isoformat(),
        })


def main() -> None:
    args = parse_args()
    if args.start_year < 1992:
        raise SystemExit("--start-year cannot precede the panel's 1992 observation start")
    client = VintageClient(os.getenv("FRED_API_KEY", ""))
    lower = date(args.start_year, 1, 1)
    upper = date(args.end_year, 12, 31)

    frames: list[pd.DataFrame] = []
    provenance_series: dict[str, dict] = {}
    total_requests = 0
    for context_id, fred_id in sorted(EXTERNAL_SERIES.items()):
        archived = [v for v in client.fetch_vintage_dates(fred_id) if lower <= v <= upper]
        # Last archived vintage per calendar month (the MRTS sampling rule).
        by_month: dict[tuple[int, int], date] = {}
        for vintage in archived:
            by_month[(vintage.year, vintage.month)] = vintage
        selected = sorted(by_month.values())
        if not selected:
            raise SystemExit(
                f"no ALFRED vintage dates in range for {context_id} ({fred_id}); "
                "external ingestion cannot proceed honestly"
            )
        total_requests += len(selected)
        provenance_series[context_id] = {
            "fred_id": fred_id,
            "vintage_count": len(selected),
            "vintage_first": selected[0].isoformat(),
            "vintage_last": selected[-1].isoformat(),
        }
        print(f"  {context_id} ({fred_id}): {len(selected)} monthly vintages")
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
        "total_vintage_fetches": total_requests,
        "sampling_rule": "last archived vintage per calendar month",
        "satellite_code_revision": git_revision(),
        "generated_at": ingested_at.isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "h7_external_ingestion.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    print(f"Wrote {len(result):,} versioned observations to {args.output.relative_to(ROOT)}")
    print(f"SHA-256 {digest} recorded in results/h7_external_ingestion.json")


if __name__ == "__main__":
    main()
