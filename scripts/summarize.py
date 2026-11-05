#!/usr/bin/env python3
"""Aggregate every candidate result into one comparison table (exploration only).

Writes results/hypothesis_table.md — the per-hypothesis table required by the
investigation brief. Never reads holdout results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, provenance, write_result


def load_all() -> list[dict]:
    rows = []

    def collect(name: str, path: Path, windows: dict) -> None:
        for window in ("choose", "report", "full_exploration"):
            entry = windows.get(window)
            if not entry:
                continue
            comparison = entry.get("comparison") or {}
            naive_wape = entry.get("naive", {}).get("mean_category_wape")
            delta = comparison.get("mean_category_wape_delta_pp")
            cand_wape = entry.get("candidate", {}).get("mean_category_wape")
            if cand_wape is None and naive_wape is not None and delta is not None:
                cand_wape = naive_wape - delta / 100
            rows.append({
                "candidate": name,
                "result_file": path.name,
                "window": window,
                "folds": entry.get("folds", entry.get("candidate", {}).get("fold_rows")),
                "naive_wape": naive_wape,
                "cand_wape": cand_wape,
                "delta_pp": delta,
                "improved": comparison.get("improved_series_count"),
                "n_series": comparison.get("series_count"),
            })

    paths = (
        sorted(RESULTS.glob("cand_*.json"))
        + sorted(RESULTS.glob("*_summary.json"))
        + sorted(RESULTS.glob("h3b_*.json"))
    )
    for path in paths:
        data = json.loads(path.read_text())
        if path.name.startswith("cand_"):
            collect(data.get("candidate", path.stem), path, data.get("windows", {}))
        elif data.get("hypothesis") == "H1":
            collect(
                f"h1 {data.get('chosen', '')} (chosen)",
                path,
                data.get("windows", {}),
            )
        elif data.get("hypothesis") == "H3":
            for label, entry in data.get("results", {}).items():
                for variant_name, variant in entry.get("variants", {}).items():
                    collect(f"h3 {label}/{variant_name}", path, variant.get("windows", {}))
        elif data.get("hypothesis") == "H3b":
            for label, entry in data.get("results", {}).items():
                collect(f"h3b {label}", path, entry.get("windows", {}))
    return rows


def main() -> None:
    rows = load_all()
    frame = pd.DataFrame(rows)
    report = frame[frame["window"] == "report"].sort_values("delta_pp", ascending=False)
    lines = [
        "# Hypothesis results — exploration origins (report window: 2021-01..2022-10)",
        "",
        "Report-window mean-category WAPE delta vs seasonal naive (positive = candidate",
        "better), with the choose window (2017-10..2020-12) shown for selection honesty.",
        "",
        "| candidate | folds | naive WAPE | candidate WAPE | delta (pp) | improved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in report.iterrows():
        cand_wape = row["cand_wape"]
        if cand_wape is None and row["naive_wape"] is not None and row["delta_pp"] is not None:
            cand_wape = row["naive_wape"] - row["delta_pp"] / 100
        lines.append(
            f"| {row['candidate']} | {row['folds']} | "
            f"{100 * row['naive_wape']:.2f}% | "
            f"{100 * cand_wape:.2f}% | {row['delta_pp']:+.2f} | "
            f"{row['improved']}/{row['n_series']} |"
        )
    lines += ["", "Provenance: " + json.dumps(provenance()["data_sha256"])]
    out = RESULTS / "hypothesis_table.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
