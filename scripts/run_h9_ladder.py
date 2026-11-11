#!/usr/bin/env python3
"""Run one H9 ladder rung on exploration origins; decide by the choose-window rule.

Protocol: H9_FEATURE_REVISIT.md (frozen before any run or pull). Boundaries
enforced here, per its §§1, 6, 7:

- Origins: the exploration grid only; the origin set is asserted equal to
  the frozen exploration grid, and any holdout origin is refused.
- Decisions use the CHOOSE window (origins < 2021-01-01) only. Report-window
  fold rows are stored but never aggregated by this script — the report
  window is scored once, by scripts/score_h9.py, after the ladder concludes.
- Advancement rule vs the current best (--compare-to-file, a folds CSV):
  (a) choose-window mean-category WAPE improvement > 0; (b) paired
  per-origin improvement on >= 9 of 15 choose origins; (c) no category
  regresses by > 3pp. Identical to the H7 rule, verbatim.
- The anchor rung (--h9-groups none --verify-against
  h7_folds_direct_growth.csv) must reproduce the H7 survivor's folds at
  max |Δ| <= 1e-6 on every exploration origin — predeclared gate 7.1; a
  mismatch aborts the experiment.
- R2's alignment mechanism must be exercised: alignment_diff_count > 0 in
  the aggregated stats — predeclared gate 7.5; a degenerate mechanism aborts.
- Data panels are hash-checked against their ingestion records; mismatch
  aborts. Missing panels abort — no synthetic substitution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    HOLDOUT_MIN_ORIGIN,
    RESULTS,
    ROOT,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
    wape,
    write_result,
)

from vintageforensics.h9 import H9_TICKERS, H9DirectLightGBM
from vintageforensics.harness import vintage_panel_backtest

H9_EXTERNAL_PANEL_PATH = ROOT / "data" / "alfred_external_vintages_h9.csv"
H9_EXTERNAL_INGESTION_RECORD = RESULTS / "h9_external_ingestion.json"
H7_EXTERNAL_PANEL_PATH = ROOT / "data" / "alfred_external_vintages.csv"
H7_EXTERNAL_INGESTION_RECORD = RESULTS / "h7_external_ingestion.json"
MARKET_PANEL_PATH = ROOT / "data" / "market_daily_closes.csv"
MARKET_INGESTION_RECORD = RESULTS / "h9_market_ingestion.json"
THETA_FOLDS = RESULTS / "folds_h5_theta.csv"
ADVANCE_MIN_ORIGINS = 9
ADVANCE_MAX_CATEGORY_REGRESSION_PP = 3.0
REPRODUCTION_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h9-groups", required=True,
                        help="comma-separated predeclared H9 groups, or 'none' for the anchor "
                             "(e.g. h9macro_const | h9macro_row | h9macro_row,h9market)")
    parser.add_argument("--out", required=True, help="results stem, e.g. anchor, r1_macro_const")
    parser.add_argument("--compare-to-file", default=None,
                        help="folds CSV of the current best (e.g. h7_folds_direct_growth.csv); "
                             "omit for the ladder anchor")
    parser.add_argument("--verify-against", default=None,
                        help="folds CSV the anchor must reproduce at 1e-6 (gate 7.1)")
    return parser.parse_args()


def _hash_checked_panel(path: Path, record_path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not record_path.exists():
        raise SystemExit(f"{label} panel not ingested (missing file or record); no substitute is permitted")
    recorded = json.loads(record_path.read_text())["output_sha256"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != recorded:
        raise SystemExit(f"{label} panel hash mismatch: {digest} != {recorded}")
    frame = pd.read_csv(path)
    frame["observation_time"] = pd.to_datetime(frame["observation_time"])
    frame["vintage_time"] = pd.to_datetime(frame["vintage_time"], utc=True)
    return frame


def per_origin_mean_category(folds: pd.DataFrame) -> dict[str, float]:
    return {
        str(origin): float(
            pd.Series(
                {
                    series: wape(group["actual"], group["prediction"])
                    for series, group in frame.groupby("series_id", sort=True)
                }
            ).mean()
        )
        for origin, frame in folds.groupby("forecast_origin", sort=True)
    }


def per_category_wape(folds: pd.DataFrame) -> dict[str, float]:
    return {
        str(series): wape(group["actual"], group["prediction"])
        for series, group in folds.groupby("series_id", sort=True)
    }


def main() -> None:
    args = parse_args()
    h9_groups = tuple(part.strip() for part in args.h9_groups.split(",") if part.strip() and part.strip() != "none")
    allowed = {"h9macro_const", "h9macro_row", "h9market"}
    unknown = set(h9_groups) - allowed
    if unknown:
        raise SystemExit(f"unknown H9 groups: {sorted(unknown)}")
    if "h9macro_const" in h9_groups and "h9macro_row" in h9_groups:
        raise SystemExit("h9macro_row replaces h9macro_const; they are never stacked")
    if h9_groups and "h9market" in h9_groups and h9_groups[-1] != "h9market":
        raise SystemExit("the predeclared ladder order ends with h9market")

    panel = load_panel()
    external_h9 = (
        _hash_checked_panel(H9_EXTERNAL_PANEL_PATH, H9_EXTERNAL_INGESTION_RECORD, "H9 external")
        if "h9macro_const" in h9_groups or "h9macro_row" in h9_groups
        else None
    )
    market = _hash_checked_panel(MARKET_PANEL_PATH, MARKET_INGESTION_RECORD, "market") if "h9market" in h9_groups else None
    # The H7 external file's SHA is re-asserted on every run (predeclared §5).
    if H7_EXTERNAL_PANEL_PATH.exists():
        recorded = json.loads(H7_EXTERNAL_INGESTION_RECORD.read_text())["output_sha256"]
        digest = hashlib.sha256(H7_EXTERNAL_PANEL_PATH.read_bytes()).hexdigest()
        if digest != recorded:
            raise SystemExit(f"H7 external panel hash changed: {digest} != {recorded}")

    instances: list[H9DirectLightGBM] = []

    def factory() -> H9DirectLightGBM:
        model = H9DirectLightGBM(
            version=f"h9_{'_'.join(h9_groups) if h9_groups else 'anchor'}",
            h9_groups=h9_groups,
            external_h9_panel=external_h9,
            market_panel=market,
        )
        if h9_groups:
            instances.append(model)
        return model

    outcome = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_end=HOLDOUT_MIN_ORIGIN,
        origin_hook=lambda model, origin: model.set_origin_context(origin),
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    if origins.max() >= HOLDOUT_MIN_ORIGIN:
        raise SystemExit("holdout origin leaked into the H9 runner")

    theta_folds = pd.read_csv(THETA_FOLDS, parse_dates=["forecast_origin", "target_time"])
    theta_exploration = theta_folds.loc[theta_folds["forecast_origin"] < HOLDOUT_MIN_ORIGIN]
    frozen_grid = pd.DatetimeIndex(sorted(theta_exploration["forecast_origin"].unique()))
    if not origins.equals(frozen_grid):
        raise SystemExit("origin grid differs from the frozen exploration grid")

    # Gate 7.1: anchor reproduction of the H7 survivor, every origin, 1e-6.
    verification = None
    if args.verify_against:
        reference = pd.read_csv(RESULTS / args.verify_against,
                                parse_dates=["forecast_origin", "target_time"])
        merged = folds.merge(
            reference,
            on=["series_id", "forecast_origin", "target_time", "horizon"],
            suffixes=("_h9", "_h7"), how="outer", indicator=True,
        )
        if not (merged["_merge"] == "both").all():
            raise SystemExit("anchor reproduction: fold keys do not match the H7 survivor")
        max_delta = float((merged["prediction_h9"] - merged["prediction_h7"]).abs().max())
        if max_delta > REPRODUCTION_TOLERANCE:
            raise SystemExit(
                f"anchor reproduction failed: max |Δ| = {max_delta} > {REPRODUCTION_TOLERANCE} — "
                "predeclared gate 7.1 stops the experiment"
            )
        verification = {
            "reference": args.verify_against,
            "max_abs_delta": max_delta,
            "tolerance": REPRODUCTION_TOLERANCE,
            "folds_compared": int(len(merged)),
            "pass": True,
        }

    # Gate 7.5: the per-row alignment mechanism must be exercised.
    alignment_stats = {"diff_count": 0, "examples": []}
    for instance in instances:
        alignment_stats["diff_count"] += instance._alignment_stats.get("diff_count", 0)
        for example in instance._alignment_stats.get("examples", []):
            if len(alignment_stats["examples"]) < 15:
                alignment_stats["examples"].append(example)
    if "h9macro_row" in h9_groups and alignment_stats["diff_count"] == 0:
        raise SystemExit(
            "alignment mechanism is degenerate (zero exercised differences) — "
            "predeclared gate 7.5 stops the experiment"
        )

    naive = naive_reference_for(origins)
    assert set(naive["forecast_origin"]) == set(origins), "naive origin mismatch"

    choose = pd.to_datetime(folds["forecast_origin"]) < CHOOSE_END
    choose_folds = folds.loc[choose]
    naive_choose = naive.loc[pd.to_datetime(naive["forecast_origin"]) < CHOOSE_END]
    theta_choose = theta_exploration.loc[pd.to_datetime(theta_exploration["forecast_origin"]) < CHOOSE_END]

    candidate_choose = evaluate_folds(choose_folds)
    naive_choose_eval = evaluate_folds(naive_choose)
    theta_choose_eval = evaluate_folds(theta_choose)
    per_origin = per_origin_mean_category(choose_folds)
    naive_per_origin = per_origin_mean_category(naive_choose)
    theta_per_origin = per_origin_mean_category(theta_choose)

    verdict_note = "ladder anchor; no compare-to target"
    advancement = None
    if args.compare_to_file:
        compare_folds = pd.read_csv(RESULTS / args.compare_to_file,
                                    parse_dates=["forecast_origin", "target_time"])
        compare_choose = compare_folds.loc[pd.to_datetime(compare_folds["forecast_origin"]) < CHOOSE_END]
        if set(pd.DatetimeIndex(sorted(compare_choose["forecast_origin"].unique()))) != set(origins[origins < CHOOSE_END]):
            raise SystemExit("compare-to config does not cover the choose-window grid")
        compare_eval = evaluate_folds(compare_choose)
        compare_per_origin = per_origin_mean_category(compare_choose)
        compare_category = per_category_wape(compare_choose)
        candidate_category = per_category_wape(choose_folds)
        delta_mean_cat = 100 * (compare_eval["mean_category_wape"] - candidate_choose["mean_category_wape"])
        improved_origins = sum(
            1 for origin in per_origin if compare_per_origin[origin] > per_origin[origin]
        )
        n_choose_origins = len(per_origin)
        category_regressions_pp = {
            series: 100 * (value - compare_category[series])
            for series, value in candidate_category.items()
            if value > compare_category[series]
        }
        rule_a = delta_mean_cat > 0
        rule_b = improved_origins >= ADVANCE_MIN_ORIGINS
        rule_c = all(v <= ADVANCE_MAX_CATEGORY_REGRESSION_PP for v in category_regressions_pp.values())
        advancement = {
            "compare_to_file": args.compare_to_file,
            "n_choose_origins": n_choose_origins,
            "mean_category_delta_pp": round(delta_mean_cat, 4),
            "improved_origins": improved_origins,
            "improved_origin_share": round(improved_origins / n_choose_origins, 4),
            "category_regressions_pp": {s: round(v, 4) for s, v in sorted(category_regressions_pp.items())},
            "rule_a_positive_delta": bool(rule_a),
            "rule_b_stable_origins": bool(rule_b),
            "rule_c_no_category_regression": bool(rule_c),
            "advance": bool(rule_a and rule_b and rule_c),
        }
        verdict_note = "ADVANCE" if advancement["advance"] else "NO-ADVANCE"

    payload = {
        "provenance": provenance({"purpose": f"H9 ladder rung {args.out} (h9: {','.join(h9_groups) or 'anchor'})"}),
        "rung": args.out,
        "h9_groups": list(h9_groups),
        "tickers": list(H9_TICKERS),
        "candidate_metadata": H9DirectLightGBM(
            version=f"h9_{'_'.join(h9_groups) if h9_groups else 'anchor'}",
            h9_groups=h9_groups,
            external_h9_panel=external_h9,
            market_panel=market,
        ).metadata(),
        "origin_window": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
            "n_origins": len(origins),
        },
        "excluded_origins": outcome.excluded_origins,
        "verification_gate_7_1": verification,
        "alignment_gate_7_5": {
            "diff_count": alignment_stats["diff_count"],
            "examples": alignment_stats["examples"],
            "required": "diff_count > 0 when h9macro_row is active",
        } if "h9macro_row" in h9_groups else None,
        "choose_window": {
            "origins": int(pd.to_datetime(choose_folds["forecast_origin"]).nunique()),
            "candidate": candidate_choose,
            "naive": naive_choose_eval,
            "theta": theta_choose_eval,
            "candidate_vs_naive": compare_to_naive(candidate_choose, naive_choose_eval),
            "candidate_vs_theta": compare_to_naive(candidate_choose, theta_choose_eval),
            "candidate_by_horizon_wape": evaluate_by_horizon(choose_folds),
            "per_origin_mean_category_wape": per_origin,
            "naive_per_origin_mean_category_wape": naive_per_origin,
            "theta_per_origin_mean_category_wape": theta_per_origin,
        },
        "advancement": advancement,
        "verdict_note": verdict_note,
        "report_window": "fold rows stored but not aggregated during selection (predeclared §6)",
    }
    write_result(f"h9_rung_{args.out}", payload)
    folds.to_csv(RESULTS / f"h9_folds_{args.out}.csv", index=False)

    print(f"{args.out}: {len(folds)} folds on {len(origins)} exploration origins, "
          f"{len(outcome.excluded_origins)} excluded origins")
    if verification:
        print(f"gate 7.1: reproduced {args.verify_against} at max |Δ| = {verification['max_abs_delta']:.2e}")
    if "h9macro_row" in h9_groups:
        print(f"gate 7.5: alignment exercised on {alignment_stats['diff_count']} (series, month) cases")
    cw = candidate_choose
    print(f"choose window: mean-cat WAPE candidate {cw['mean_category_wape']:.4f} | "
          f"naive {naive_choose_eval['mean_category_wape']:.4f} | "
          f"theta {theta_choose_eval['mean_category_wape']:.4f} | "
          f"pooled candidate {cw['aggregate_wape']:.4f}")
    if advancement:
        print(f"vs {args.compare_to_file}: {advancement['mean_category_delta_pp']:+.2f}pp, "
              f"{advancement['improved_origins']}/{advancement['n_choose_origins']} origins improved, "
              f"category regressions {advancement['category_regressions_pp'] or 'none'} -> {verdict_note}")


if __name__ == "__main__":
    main()
