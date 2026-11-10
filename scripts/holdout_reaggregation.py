"""Three-model holdout table: reporting-only reaggregation of stored folds.

Predeclared before recalculation (2026-11-10), committed before the numbers:

1. The promotion rule is unchanged: mean-category WAPE improvement,
   >=10/11 categories improved, bounded regressions. Nothing here re-scores
   any candidate against a different rule.
2. No model will be tuned, re-selected, or re-fitted based on this
   comparison.
3. This is a reporting-only reaggregation of STORED gate-reference fold
   predictions (``reference/vintage_backtest_folds.csv``). No prediction is
   regenerated; theta and its naive reference stay verbatim from the frozen
   single run (``results/holdout_theta.json``).
4. The LightGBM result is labeled "previously observed (gate reference
   folds)" - it is NOT an untouched holdout result. Those origins entered
   the record through the original gate run, before the window was reserved
   as holdout.
5. Verification gate: the reaggregated naive numbers must reproduce
   ``holdout_theta.json``'s naive block exactly (same folds, same metric),
   and the LightGBM mean-category WAPE must reproduce the recorded 5.64%
   gate slice. Any discrepancy STOPS this table until investigated; the
   more favorable number is never chosen.

Rationale (why this is legitimate while a fourth look would not be): what
invalidates a holdout is using its results to modify models, add candidates,
or change the decision rule. LightGBM is the incumbent champion, not a new
candidate; its holdout-origin predictions already exist in a frozen artifact;
reaggregating them creates no new information and changes no decision.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vintageforensics.metrics import wape  # noqa: E402

FOLDS = ROOT / "reference" / "vintage_backtest_folds.csv"
HOLDOUT_RUN = ROOT / "results" / "holdout_theta.json"
OUTPUT = ROOT / "results" / "holdout_three_model.json"

PREDECLARED_STATEMENT = (
    "Reporting-only reaggregation, predeclared 2026-11-10 before recalculation: "
    "promotion rule unchanged; no tuning or re-selection; stored gate-reference "
    "folds only, no prediction regenerated; LightGBM labeled 'previously observed "
    "(gate reference folds)', not an untouched holdout result; naive/theta remain "
    "verbatim from the frozen single run."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    holdout_run = json.loads(HOLDOUT_RUN.read_text())
    origins = holdout_run["provenance"]["holdout_origins"]
    folds = pd.read_csv(FOLDS)
    sub = folds[folds["forecast_origin"].isin(origins)]
    if sub.groupby("model_name").size().to_dict() != {
        "global_lightgbm": 352,
        "seasonal_naive": 352,
    }:
        sys.exit(
            "STOP: stored folds do not contain exactly 352 rows per model on the "
            "holdout origins - investigate before releasing any table."
        )

    def per_series(model: str) -> dict[str, float]:
        rows = sub[sub["model_name"] == model]
        return {
            series: wape(group["actual"], group["prediction"])
            for series, group in rows.groupby("series_id")
        }

    lgbm = per_series("global_lightgbm")
    naive_folds = per_series("seasonal_naive")

    def mean_category(by_series: dict[str, float]) -> float:
        return sum(by_series.values()) / len(by_series)

    lgbm_mean = mean_category(lgbm)
    naive_mean = mean_category(naive_folds)

    # Verification gate (predeclared item 5).
    ref_naive_mean = holdout_run["naive"]["mean_category_wape"]
    if abs(naive_mean - ref_naive_mean) > 1e-12:
        sys.exit(
            f"STOP: reaggregated naive mean-category WAPE {naive_mean:.6f} does not "
            f"reproduce the frozen run's {ref_naive_mean:.6f}. Investigate before "
            "releasing this table."
        )
    recorded_slice = 0.0564  # FINDINGS section 3, two decimals
    if round(lgbm_mean * 100, 2) != round(recorded_slice * 100, 2):
        sys.exit(
            f"STOP: reaggregated LightGBM mean-category WAPE {lgbm_mean * 100:.4f}% "
            "does not reproduce the recorded 5.64% gate slice. Investigate before "
            "releasing this table."
        )

    theta_by_series = {
        s: m["wape"] for s, m in holdout_run["candidate"]["by_series"].items()
    }
    theta_mean = holdout_run["candidate"]["mean_category_wape"]
    categories = sorted(lgbm)
    rows = []
    for name, by_series, mean_value, source in (
        ("seasonal_naive", naive_folds, naive_mean, "results/holdout_theta.json (frozen single run)"),
        ("theta_method", theta_by_series, theta_mean, "results/holdout_theta.json (frozen single run)"),
        (
            "global_lightgbm",
            lgbm,
            lgbm_mean,
            "reference/vintage_backtest_folds.csv (gate slice; previously observed, not an untouched holdout result)",
        ),
    ):
        delta_pp = (naive_mean - mean_value) * 100
        rows.append(
            {
                "model_name": name,
                "mean_category_wape": mean_value,
                "delta_pp_vs_naive": round(delta_pp, 2),
                "categories_beating_naive": sum(
                    1 for s in categories if by_series[s] < naive_folds[s]
                ),
                "categories_total": len(categories),
                "by_series_wape": by_series,
                "source_artifact": source,
            }
        )

    output = {
        "provenance": {
            "predeclared_statement": PREDECLARED_STATEMENT,
            "folds_source": "reference/vintage_backtest_folds.csv",
            "folds_sha256": _sha256(FOLDS),
            "gate_code_revision_reference": holdout_run["provenance"][
                "gate_code_revision_reference"
            ],
            "data_sha256": holdout_run["provenance"]["data_sha256"],
            "holdout_origins": origins,
            "verification": {
                "naive_reproduces_frozen_run": True,
                "lightgbm_matches_recorded_slice_pct": round(lgbm_mean * 100, 2),
            },
        },
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=1) + "\n")

    print("verification: naive reproduces the frozen run; "
          f"LightGBM slice = {lgbm_mean * 100:.2f}%")
    for row in rows:
        print(
            f"{row['model_name']:<18} {row['mean_category_wape'] * 100:5.2f}%  "
            f"{row['delta_pp_vs_naive']:+.2f}pp  "
            f"{row['categories_beating_naive']}/{row['categories_total']} categories"
        )


if __name__ == "__main__":
    main()
