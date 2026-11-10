#!/usr/bin/env python3
"""Run one H7 ladder rung on exploration origins; decide by the choose-window rule.

Protocol: H7_FEATURE_LADDER.md (frozen before any run). Boundaries enforced
here, per its §§1, 6, 7:

- Origins: the exploration grid only (origin_end = HOLDOUT_MIN_ORIGIN);
  the origin set is asserted equal to the frozen exploration grid, and any
  holdout origin is refused.
- Decisions use the CHOOSE window (origins < 2021-01-01) only. Report-window
  fold rows are stored but never aggregated by this script — the report
  window is scored once, by scripts/score_h7.py, after the ladder concludes.
- Advancement rule vs the current best (--compare-to): (a) choose-window
  mean-category WAPE improvement > 0; (b) paired per-origin improvement on
  >= 9 of 15 choose origins; (c) no category regresses by > 3pp. The first
  rung of the ladder has no compare-to target.
- Frozen benchmarks (naive, theta) are computed on the choose window for
  context only; they never advance a rung.
- External rung: requires data/alfred_external_vintages.csv whose SHA-256
  matches results/h7_external_ingestion.json; mismatch aborts.
- Excluded origins are recorded with reasons, never silently dropped.
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

from vintageforensics.direct import DirectElasticNet, DirectLightGBM
from vintageforensics.harness import vintage_panel_backtest

EXTERNAL_PANEL_PATH = ROOT / "data" / "alfred_external_vintages.csv"
EXTERNAL_INGESTION_RECORD = RESULTS / "h7_external_ingestion.json"
THETA_FOLDS = RESULTS / "folds_h5_theta.csv"
ADVANCE_MIN_ORIGINS = 9
ADVANCE_MAX_CATEGORY_REGRESSION_PP = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lgbm", "elastic"), required=True)
    parser.add_argument("--groups", required=True,
                        help="comma-separated predeclared groups, base first, e.g. base,growth")
    parser.add_argument("--out", required=True, help="results stem, e.g. direct_base")
    parser.add_argument("--compare-to", default=None,
                        help="results stem of the current best (folds CSV); omit for the ladder anchor")
    return parser.parse_args()


def load_external_panel() -> pd.DataFrame:
    if not EXTERNAL_PANEL_PATH.exists() or not EXTERNAL_INGESTION_RECORD.exists():
        raise SystemExit(
            "external panel not ingested: run scripts/fetch_external_vintages.py "
            "(missing file or ingestion record); no substitute is permitted"
        )
    recorded = json.loads(EXTERNAL_INGESTION_RECORD.read_text())["output_sha256"]
    digest = hashlib.sha256(EXTERNAL_PANEL_PATH.read_bytes()).hexdigest()
    if digest != recorded:
        raise SystemExit(f"external panel hash mismatch: {digest} != {recorded}")
    frame = pd.read_csv(EXTERNAL_PANEL_PATH)
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
    groups = tuple(part.strip() for part in args.groups.split(",") if part.strip())
    if not groups or groups[0] != "base":
        raise SystemExit("groups must start with 'base' (the ladder's anchor)")
    unknown = set(groups) - {"base", "growth", "trend", "cross", "revision", "external"}
    if unknown:
        raise SystemExit(f"unknown groups: {sorted(unknown)}")

    panel = load_panel()
    external_panel = load_external_panel() if "external" in groups else None

    def factory():
        model_cls = DirectLightGBM if args.model == "lgbm" else DirectElasticNet
        version = f"h7_{args.model}_{'_'.join(groups)}"
        return model_cls(
            version=version,
            groups=groups,
            vintage_panel=panel if "revision" in groups else None,
            external_panel=external_panel,
        )

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
        raise SystemExit("holdout origin leaked into the H7 runner")

    theta_folds = pd.read_csv(THETA_FOLDS, parse_dates=["forecast_origin", "target_time"])
    theta_exploration = theta_folds.loc[theta_folds["forecast_origin"] < HOLDOUT_MIN_ORIGIN]
    frozen_grid = pd.DatetimeIndex(sorted(theta_exploration["forecast_origin"].unique()))
    if not origins.equals(frozen_grid):
        raise SystemExit("origin grid differs from the frozen exploration grid")

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
    if args.compare_to:
        compare_folds = pd.read_csv(RESULTS / f"h7_folds_{args.compare_to}.csv",
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
            "compare_to": args.compare_to,
            "n_choose_origins": n_choose_origins,
            "mean_category_delta_pp": round(delta_mean_cat, 4),
            "improved_origins": improved_origins,
            "improved_origin_share": round(improved_origins / n_choose_origins, 4),
            "category_regressions_pp": {s: round(v, 4) for s, v in sorted(category_regressions_pp.items())},
            "rule_a_positive_delta": bool(rule_a),
            "rule_b_stable_origins": bool(rule_b),
            "rule_c_no_category_regression": bool(rule_c),
            "advance": bool(rule_a and rule_b and rule_c),
            "required_improved_origins": ADVANCE_MIN_ORIGINS,
            "max_category_regression_pp_allowed": ADVANCE_MAX_CATEGORY_REGRESSION_PP,
        }
        verdict_note = "ADVANCE" if advancement["advance"] else "NO-ADVANCE"

    payload = {
        "provenance": provenance({"purpose": f"H7 ladder rung {args.out} ({args.model}: {','.join(groups)})"}),
        "rung": args.out,
        "model": args.model,
        "groups": list(groups),
        "candidate_metadata": factory().metadata(),
        "origin_window": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
            "n_origins": len(origins),
        },
        "excluded_origins": outcome.excluded_origins,
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
    write_result(f"h7_rung_{args.out}", payload)
    folds_out = RESULTS / f"h7_folds_{args.out}.csv"
    folds.to_csv(folds_out, index=False)

    print(f"{args.out}: {len(folds)} folds on {len(origins)} exploration origins, "
          f"{len(outcome.excluded_origins)} excluded origins")
    cw = candidate_choose
    print(f"choose window: mean-cat WAPE candidate {cw['mean_category_wape']:.4f} | "
          f"naive {naive_choose_eval['mean_category_wape']:.4f} | "
          f"theta {theta_choose_eval['mean_category_wape']:.4f} | "
          f"pooled candidate {cw['aggregate_wape']:.4f}")
    if advancement:
        print(f"vs {args.compare_to}: {advancement['mean_category_delta_pp']:+.2f}pp, "
              f"{advancement['improved_origins']}/{advancement['n_choose_origins']} origins improved, "
              f"category regressions {advancement['category_regressions_pp'] or 'none'} -> {verdict_note}")


if __name__ == "__main__":
    main()
