#!/usr/bin/env python3
"""Recover theta's holdout fold rows — deterministic re-derivation, verification-locked.

Predeclared before execution (2026-11-10) and committed before this script ran:

1. Scope: artifact recovery ONLY. The single holdout run
   (``results/holdout_theta.json``, 2026-11-05) recorded aggregate metrics but
   not its fold rows, so uncertainty quantification (origin-block CIs), MASE,
   signed bias, category-by-horizon, and first-release re-scoring of the
   holdout headline are impossible from stored artifacts. This script
   re-derives those fold rows through the identical library path and protocol
   as ``scripts/holdout.py``. The computation is deterministic: theta has no
   random component, and the data hash and pinned environment are recorded
   below.
2. No decision input. The promotion verdict is ``holdout_theta.json``'s and
   stays as recorded (criterion 1 passes, criterion 2 fails at 7/11,
   criterion 3 passes). Nothing downstream of this recovery feeds back into
   any model, selection, or rule; the scorecard it enables is reporting-only
   (``scripts/holdout_scorecard.py``).
3. Verification gate (stops everything on mismatch): the recomputed theta
   aggregates must reproduce the frozen run's numbers — aggregate WAPE,
   mean-category WAPE, every by-series WAPE, and every by-horizon WAPE, each
   to 1e-9. The naive reference from the stored gate folds must reproduce the
   frozen naive block to 1e-12. The fold count must be 352 and the origins
   exactly the frozen eight. A mismatch means STOP and investigate — never a
   rerun with adjusted parameters. The more favorable number is never chosen.
4. Relationship to the previous bright line: ``holdout_reaggregation.py``
   drew the line "no prediction is regenerated" when it re-aggregated stored
   LightGBM folds. This recovery crosses that literal line for theta under
   the owner's explicit 2026-11-10 decision, because a verification-locked
   re-derivation of a deterministic computation creates no new information
   about the holdout: it either reproduces the frozen numbers bit-for-bit
   (to float tolerance) or it stops. What would actually invalidate the
   holdout — new candidates, changed rules, tuned models, selection on
   holdout results — remains forbidden.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HOLDOUT_MIN_ORIGIN,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
    write_result,
)
from vintageforensics.candidates import ThetaMethod  # noqa: E402
from vintageforensics.harness import vintage_panel_backtest  # noqa: E402
from vintageforensics.models import PerSeriesPanelAdapter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN_RUN = ROOT / "results" / "holdout_theta.json"
SCHEMA_REFERENCE = ROOT / "results" / "folds_h5_theta.csv"
RECOVERED_CSV = ROOT / "results" / "folds_holdout_theta.csv"

AGG_TOL = 1e-9
NAIVE_TOL = 1e-12

PREDECLARED_STATEMENT = (
    "Deterministic recovery of theta holdout fold rows, predeclared 2026-11-10 "
    "and committed before execution: identical library path and protocol to the "
    "frozen single run; theta and naive aggregates must reproduce "
    "results/holdout_theta.json exactly (1e-9 / 1e-12) or the script stops; "
    "no model, selection, or rule changes; promotion verdict unchanged; "
    "recovered rows feed the reporting-only scorecard, nothing else."
)


def _close(recomputed: float, frozen: float, tol: float, label: str, diffs: dict) -> None:
    diff = abs(recomputed - frozen)
    if diff > tol:
        sys.exit(
            f"STOP: {label} recomputed {recomputed!r} vs frozen {frozen!r} "
            f"(diff {diff:.3e} > {tol:.0e}). Investigate before releasing any "
            "recovered artifact."
        )
    diffs[label] = diff


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frozen = json.loads(FROZEN_RUN.read_text())
    frozen_origins = [pd.Timestamp(o) for o in frozen["holdout_origins"]]

    panel = load_panel()
    outcome = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(ThetaMethod),
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=HOLDOUT_MIN_ORIGIN,
    )
    folds = outcome.folds

    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    if len(folds) != 352:
        sys.exit(f"STOP: recovered {len(folds)} folds, expected the frozen 352.")
    if [o.date().isoformat() for o in origins] != frozen["holdout_origins"]:
        sys.exit(
            "STOP: recovered origins "
            f"{[o.date().isoformat() for o in origins]} do not equal the frozen "
            f"{frozen['holdout_origins']}."
        )
    if origins.min() < HOLDOUT_MIN_ORIGIN:
        sys.exit("STOP: exploration origin leaked into the recovery run.")

    theta_eval = evaluate_folds(folds)
    theta_hz = evaluate_by_horizon(folds)
    naive = naive_reference_for(origins)
    if set(naive["forecast_origin"]) != set(origins):
        sys.exit("STOP: naive origin mismatch on holdout recovery.")
    naive_eval = evaluate_folds(naive)

    diffs: dict[str, float] = {}
    for stat in ("aggregate_wape", "mean_category_wape"):
        _close(theta_eval[stat], frozen["candidate"][stat], AGG_TOL, f"theta.{stat}", diffs)
        _close(naive_eval[stat], frozen["naive"][stat], NAIVE_TOL, f"naive.{stat}", diffs)
    for series, block in frozen["candidate"]["by_series"].items():
        _close(theta_eval["by_series"][series]["wape"], block["wape"], AGG_TOL,
               f"theta.by_series.{series}", diffs)
    for series, block in frozen["naive"]["by_series"].items():
        _close(naive_eval["by_series"][series]["wape"], block["wape"], NAIVE_TOL,
               f"naive.by_series.{series}", diffs)
    naive_hz = evaluate_by_horizon(naive)
    for horizon, value in frozen["by_horizon"]["theta"].items():
        _close(theta_hz[horizon], value, AGG_TOL, f"theta.by_horizon.{horizon}", diffs)
    for horizon, value in frozen["by_horizon"]["naive"].items():
        _close(naive_hz[horizon], value, NAIVE_TOL,
               f"naive.by_horizon.{horizon}", diffs)

    schema = pd.read_csv(SCHEMA_REFERENCE, nrows=0).columns.tolist()
    recovered = folds[schema].copy()
    if set(recovered["model_name"].unique()) != {"theta_method"}:
        sys.exit(
            f"STOP: unexpected model names {recovered['model_name'].unique()} in "
            "recovered folds."
        )
    recovered.to_csv(RECOVERED_CSV, index=False)

    write_result(
        "holdout_theta_recovery",
        {
            "provenance": provenance(
                {"purpose": "artifact recovery of theta holdout fold rows (reporting only)"}
            ),
            "predeclared_statement": PREDECLARED_STATEMENT,
            "frozen_run": "results/holdout_theta.json",
            "frozen_run_sha256_of_input_data": frozen["provenance"]["data_sha256"],
            "recovered_rows": int(len(recovered)),
            "recovered_origins": frozen["holdout_origins"],
            "recovered_folds_sha256": _sha256(RECOVERED_CSV),
            "verification": {
                "all_gates_passed": True,
                "theta_tolerance": AGG_TOL,
                "naive_tolerance": NAIVE_TOL,
                "max_theta_diff": max(v for k, v in diffs.items() if k.startswith("theta")),
                "max_naive_diff": max(v for k, v in diffs.items() if k.startswith("naive")),
                "diffs": diffs,
            },
            "scope_note": (
                "Recovered rows exist so the reporting-only scorecard (MASE, signed "
                "bias, category-by-horizon, origin-block CIs, first-release "
                "re-scoring) can be computed on the holdout. The promotion verdict "
                "remains holdout_theta.json's; nothing here feeds back into any "
                "model, selection, or rule."
            ),
        },
    )
    print(
        f"verification PASSED: all {len(diffs)} checked aggregates reproduce the "
        f"frozen single run (max theta diff {max(v for k, v in diffs.items() if k.startswith('theta')):.2e}, "
        f"max naive diff {max(v for k, v in diffs.items() if k.startswith('naive')):.2e})"
    )
    print(f"wrote {RECOVERED_CSV.relative_to(ROOT)} ({len(recovered)} fold rows)")


if __name__ == "__main__":
    main()
