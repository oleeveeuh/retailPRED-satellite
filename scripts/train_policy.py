"""Train and freeze the regime-switch policy (POLICY.md §5).

Everything here runs on CHOOSE origins only (exploration, < 2021-01): labels
from the frozen fold files, causal features from the vintage panel, OOF
logistic/LightGBM scores, isotonic calibration, tau sweep. The frozen
artifact (model + isotonic + tau) is written for the single REPORT pass in
scripts/eval_policy.py. Holdout origins (>= 2023-01) are never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
from regime.causality import assert_origin_window, assert_train_before_eval  # noqa: E402
from regime.features import FEATURE_COLUMNS, build_feature_table  # noqa: E402
from regime.labels import (  # noqa: E402
    assert_as_of_alignment,
    build_labels,
    check_actuals_consistent,
    load_fold_data,
)
from regime.model import MODEL_KINDS, blend_pair_sums, fit_policy, pair_sums_from_folds  # noqa: E402
from vintageforensics.harness import availability_as_of  # noqa: E402

BOUNDARY = common.HOLDOUT_MIN_ORIGIN
CHOOSE_END = common.CHOOSE_END
GLOBAL_THETA_WEIGHT = 0.5  # recorded: results/h3b_theta_blend.json
EXPECTED_CATEGORIES = 11
ARTIFACT_PATH = common.RESULTS / "policy_frozen.joblib"


def main() -> None:
    panel = common.load_panel()

    naive_data, naive_dropped = load_fold_data(common.REFERENCE_FOLDS, "seasonal_naive", BOUNDARY)
    theta_path = common.RESULTS / "folds_h5_theta.csv"
    theta_data, theta_dropped = load_fold_data(theta_path, "theta_method", BOUNDARY)
    assert_origin_window(theta_data.origins, BOUNDARY)

    # POLICY.md §4: prove fast selection equivalence before any feature is built.
    expected_as_of = pd.Series(
        {origin: availability_as_of(origin, common.PROTOCOL["availability_lag_days"])
         for origin in naive_data.origins}
    )
    common.assert_selection_equivalence(panel, list(expected_as_of))

    assert_as_of_alignment(naive_data.folds, expected_as_of)
    assert_as_of_alignment(theta_data.folds, expected_as_of)
    check_actuals_consistent(naive_data, theta_data)

    label_table = build_labels(naive_data, theta_data)
    table = build_feature_table(panel, naive_data.origins, naive_data, theta_data,
                                availability_as_of)
    table = table.set_index(["series_id", "forecast_origin"]).join(
        label_table.labels.rename("theta_wins")
    ).join(label_table.naive_wape).join(label_table.theta_wape).reset_index()
    if table["theta_wins"].isna().any():
        raise AssertionError("feature rows without labels")

    origins = pd.DatetimeIndex(sorted(table["forecast_origin"].unique()))
    assert_origin_window(origins, BOUNDARY)
    series_categories = sorted(table["series_id"].unique())
    if len(series_categories) != EXPECTED_CATEGORIES:
        raise AssertionError(f"expected {EXPECTED_CATEGORIES} categories, saw {len(series_categories)}")

    choose_origins = origins[origins < CHOOSE_END]
    report_origins = origins[(origins >= CHOOSE_END) & (origins < BOUNDARY)]
    assert_train_before_eval(choose_origins, report_origins)
    if report_origins.max() > pd.Timestamp("2022-10-01"):
        raise AssertionError(f"unexpected REPORT grid: max {report_origins.max()}")

    choose = table.loc[table["forecast_origin"] < CHOOSE_END].copy()
    report = table.loc[
        (table["forecast_origin"] >= CHOOSE_END) & (table["forecast_origin"] < BOUNDARY)
    ].copy()
    # POLICY.md §4.4: rows lacking published-error history are excluded from
    # training, never imputed; every evaluation row must be complete.
    train = choose.loc[~choose["error_features_pending"]].copy()
    if report["error_features_pending"].any():
        raise AssertionError("REPORT rows with incomplete features")
    if train["theta_wins"].isna().any() or report["theta_wins"].isna().any():
        raise AssertionError("missing labels after the split")

    naive_sums = pair_sums_from_folds(naive_data.folds)
    theta_sums = pair_sums_from_folds(theta_data.folds)
    blend_sums = blend_pair_sums(naive_data.folds, theta_data.folds, GLOBAL_THETA_WEIGHT)

    policies = {
        kind: fit_policy(kind, train, naive_sums, theta_sums, blend_sums, series_categories)
        for kind in MODEL_KINDS
    }
    # CHOOSE-side model choice; ties resolve to the simpler logistic form.
    chosen_kind = min(
        MODEL_KINDS,
        key=lambda k: (round(policies[k].choose_policy_wape, 9), 0 if k == "logistic" else 1),
    )
    policy = policies[chosen_kind]

    sweep_frame = policy.sweep
    sweep_records = sweep_frame.drop(columns=[], errors="ignore").to_dict("records")
    sweep_json = [
        {key: (float(value) if isinstance(value, (int, float,)) else value)
         for key, value in row.items()}
        for row in sweep_records
    ]
    label_rates = {
        "all": float(table["theta_wins"].mean()),
        "choose": float(choose["theta_wins"].mean()),
        "report": float(report["theta_wins"].mean()),
        "train_rows": float(train["theta_wins"].mean()),
    }
    payload = {
        "provenance": common.provenance({
            "n_pairs_total": int(len(table)),
            "n_pairs_choose": int(len(choose)),
            "n_pairs_report": int(len(report)),
            "n_train_rows": int(len(train)),
            "sample_size_note": (
                f"n={len(report)} REPORT pairs (9 origins x 11 categories); low-complexity "
                "models only; all uncertainties reported as origin-block bootstrap CIs"
            ),
            "fold_rows_dropped_holdout": {
                "reference_vintage_backtest_folds": naive_dropped,
                "folds_h5_theta": theta_dropped,
            },
            "global_theta_weight_blend_reference": GLOBAL_THETA_WEIGHT,
            "feature_columns": list(FEATURE_COLUMNS),
        }),
        "labels_theta_win_rate": label_rates,
        "model_choice": {
            kind: {
                "tau": policies[kind].tau,
                "oof_brier": policies[kind].oof_brier,
                "choose_policy_mean_cat_wape": policies[kind].choose_policy_wape,
            }
            for kind in MODEL_KINDS
        },
        "chosen_model": chosen_kind,
        "tau": policy.tau,
        "tau_grid_note": "0.00..1.00 step 0.02, guardrail 1 eligible set, ties -> larger tau",
        "tau_sweep_choose_oof": sweep_json,
        "tau_sweep_reference_mean_cat_wape": sweep_frame.attrs.get(
            "reference_mean_cat", {}
        ),
        "guardrail1_bounded_downside_choose_oof": {
            "policy_mean_cat_wape": policy.choose_policy_wape,
            "delta_vs_naive_pp": 100.0 * (
                policy.choose_policy_wape
                - sweep_frame.attrs["reference_mean_cat"]["naive"]
            ),
            "max_allowed_loss_pp": 0.5,
            "passes": bool(
                policy.choose_policy_wape
                <= sweep_frame.attrs["reference_mean_cat"]["naive"] + 0.005
            ),
        },
        "oof_brier_choose": policy.oof_brier,
        "train_origins": [str(o.date()) for o in policy.train_origins],
        "report_origins": [str(o.date()) for o in report_origins],
    }
    common.write_result("policy_train", payload)

    joblib.dump(
        {
            "policy": policy,
            "train_table": train,
            "series_categories": series_categories,
            "global_theta_weight": GLOBAL_THETA_WEIGHT,
            "provenance": payload["provenance"],
        },
        ARTIFACT_PATH,
    )
    print(f"wrote {ARTIFACT_PATH.relative_to(ROOT)}")
    print(f"chosen model: {chosen_kind}  tau: {policy.tau:.2f}  "
          f"CHOOSE OOF policy mean-category WAPE: {policy.choose_policy_wape:.4f}")
    print(f"train rows: {len(train)}  CHOOSE pairs: {len(choose)}  "
          f"REPORT pairs: {len(report)}  (holdout never read: "
          f"{naive_dropped} gate rows, {theta_dropped} theta rows dropped at load)")


if __name__ == "__main__":
    main()
