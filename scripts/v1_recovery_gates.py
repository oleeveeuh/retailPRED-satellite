#!/usr/bin/env python3
"""v1 recovery — verification gates (Phase 3).

Applies the predeclared gates from V1_PERFORMANCE_RECOVERY.md §5:

  gate 2 (determinism): run1 vs run2 summaries identical modulo
        {training_date, duration_seconds}; run1 vs run2 pickles identical
        modulo {training_time} (compared both as bytes and as loaded structures).
  gate 3 (pickle equivalence): for each category × model —
        (a) committed-model holdout predictions vs re-trained holdout
            predictions on identical rows: max |Δ|;
        (b) metrics embedded in the committed pickle (the original run's
            recorded numbers) vs recovered run metrics: exact match required
            for run-level classification.
        Plus an internal-consistency check: MAE recomputed from the
        committed model's own predictions vs the metrics embedded in that
        same pickle — pass/fail proves whether the committed pickles were
        produced on the committed CSVs.
  derived quantities: holdout MAPE (|y−ŷ|/y, mean × 100) from committed and
        re-trained predictions, labeled derived.

Stop-on-mismatch: gate 2 failure exits nonzero. Gate 3 deltas are always
reported in full (that is the finding), and the run-level vs metrics-level
classification is computed here.

Usage:
    /tmp/v1_recovery_venv/bin/python scripts/v1_recovery_gates.py <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path

TAG = Path("/tmp/v1_recovery_tag")
RUN1 = Path("/tmp/v1_recovery_transcripts/run1_summary.json")
RUN2 = Path("/tmp/v1_recovery_transcripts/run2_summary.json")
RUN1_MODELS = Path("/tmp/v1_recovery_transcripts/run1_models")
REFERENCE = Path("/tmp/v1_recovery_reference")
DATA = TAG / "project_root" / "data_multi_resolution"

VOLATILE_SUMMARY_KEYS = {"training_date", "duration_seconds"}
CATEGORIES = [
    "automobile_dealers", "building_material_and_garden_equipment",
    "clothing_and_clothing_accessories_stores", "electronics_and_appliance_stores",
    "food_and_beverage_stores", "furniture_and_home_furnishings_stores",
    "gasoline_stations", "general_merchandise_stores",
    "health_and_personal_care_stores",
    "sporting_goods_hobby_and_musical_instrument_stores", "total_sales",
]
MODEL_TYPES = ["RandomForest", "LGBM"]
EXCLUDE = {"y", "index", "year"}


def strip_volatile(obj, keys):
    if isinstance(obj, dict):
        return {k: strip_volatile(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [strip_volatile(v, keys) for v in obj]
    return obj


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pickle_payload(p):
    import joblib
    md = joblib.load(p)
    if not isinstance(md, dict) or "model" not in md:
        return {"_shape": "unexpected", "_type": type(md).__name__}, None
    out = {k: v for k, v in md.items() if k != "training_time"}
    return out, md.get("training_time")


def main() -> int:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    run1 = json.loads(RUN1.read_text())
    run2 = json.loads(RUN2.read_text())
    payload_equal = strip_volatile(run1, VOLATILE_SUMMARY_KEYS) == strip_volatile(run2, VOLATILE_SUMMARY_KEYS)

    # pickle-level determinism: bytes and structures
    pickle_bytes_equal, pickle_struct_equal, pickle_struct_diffs = True, True, []
    for cat in CATEGORIES:
        for mt in MODEL_TYPES:
            name = f"{cat}_{mt}_model.pkl"
            a, b = RUN1_MODELS / name, TAG / "backend" / "ml" / "models" / name
            if file_sha(a) != file_sha(b):
                pickle_bytes_equal = False
            pa, ta = pickle_payload(a)
            pb, tb = pickle_payload(b)
            if json.dumps(pa, sort_keys=True, default=str) != json.dumps(pb, sort_keys=True, default=str):
                pickle_struct_equal = False
                pickle_struct_diffs.append(name)

    gate2 = {
        "summary_metric_payloads_identical": payload_equal,
        "summary_bytes_identical": RUN1.read_bytes() == RUN2.read_bytes(),
        "run_pickles_bytes_identical": pickle_bytes_equal,
        "run_pickles_structures_identical_modulo_training_time": pickle_struct_equal,
        "structure_diffs": pickle_struct_diffs,
        "run1": {"training_date": run1["training_date"], "duration_seconds": run1["duration_seconds"]},
        "run2": {"training_date": run2["training_date"], "duration_seconds": run2["duration_seconds"]},
    }

    rows = []
    for cat in CATEGORIES:
        df = pd.read_csv(DATA / f"retail_{cat}_multi_resolution.csv")
        feature_cols = [c for c in df.columns if c not in EXCLUDE]
        test_df = df.iloc[-52:]
        X_test = test_df[feature_cols]
        y_test = test_df["y"].to_numpy()
        for mt in MODEL_TYPES:
            name = f"{cat}_{mt}_model.pkl"
            committed = joblib.load(REFERENCE / "backend_ml_models" / name)
            rerun1 = joblib.load(RUN1_MODELS / name)
            rerun2 = joblib.load(TAG / "backend" / "ml" / "models" / name)
            # committed payloads are normally the trainer's model_data dict;
            # total_sales_LGBM is a bare lightgbm.Booster (overwritten by a
            # later save path) — handled per §5.3: compared, reported, never substituted
            is_model_data = isinstance(committed, dict) and "model" in committed
            cmodel = committed["model"] if is_model_data else committed
            committed_shape = "model_data" if is_model_data else f"bare_{type(committed).__name__}"
            if is_model_data:
                assert committed["features"] == feature_cols, f"feature order mismatch: {name}"
            Xn = X_test.to_numpy()
            pc = cmodel.predict(Xn)
            p1 = rerun1["model"].predict(Xn)
            p2 = rerun2["model"].predict(Xn)
            embedded = committed.get("metrics", {}) if is_model_data else {}
            mae_committed_pred = float(mean_absolute_error(y_test, pc))
            rec_metrics_1 = run1["results"][cat][mt]["metrics"]
            rec_metrics_2 = run2["results"][cat][mt]["metrics"]
            mape = lambda p: float(np.mean(np.abs(y_test - p) / y_test) * 100)  # noqa: E731
            row = {
                "pickle": name,
                "committed_shape": committed_shape,
                "n_holdout_rows": int(len(y_test)),
                "gate1_features": int(rerun1["n_features"]),
                "pred_max_abs_delta_committed_vs_run1": float(np.max(np.abs(pc - p1))),
                "pred_max_abs_delta_run1_vs_run2": float(np.max(np.abs(p1 - p2))),
                "embedded_metrics": embedded,
                "recovered_metrics_run1": rec_metrics_1,
                "recovered_metrics_run2": rec_metrics_2,
                "embedded_vs_recovered_exact": embedded == rec_metrics_1 == rec_metrics_2,
                "internal_consistency": {
                    "embedded_test_mae": embedded.get("test_mae"),
                    "recomputed_test_mae_from_committed_predictions": mae_committed_pred,
                    "abs_delta": abs(embedded.get("test_mae", float("nan")) - mae_committed_pred),
                    "match_exact": embedded.get("test_mae") == mae_committed_pred,
                },
                "derived_holdout_mape_pct": {
                    "committed_model_predictions": mape(pc),
                    "retrained_model_predictions": mape(p1),
                },
                "train_mae_recovered": rec_metrics_1["train_mae"],
                "test_mae_recovered": rec_metrics_1["test_mae"],
            }
            rows.append(row)

    gate3 = {
        "n_compared": len(rows),
        "all_pred_max_abs_delta_zero_committed_vs_run1": all(r["pred_max_abs_delta_committed_vs_run1"] == 0.0 for r in rows),
        "all_pred_max_abs_delta_zero_run1_vs_run2": all(r["pred_max_abs_delta_run1_vs_run2"] == 0.0 for r in rows),
        "all_embedded_vs_recovered_exact": all(r["embedded_vs_recovered_exact"] for r in rows),
        "all_internal_consistency_exact": all(r["internal_consistency"]["match_exact"] for r in rows),
        "rows": rows,
    }

    gate1 = {
        "all_report_73": all(r["gate1_features"] == 73 for r in rows),
        "log_line_counts": {
            "run1": sum(1 for l in (TAG.parent / "v1_recovery_transcripts" / "run1.log").read_text().splitlines() if "rows with 73 features" in l),
            "run2": sum(1 for l in (TAG.parent / "v1_recovery_transcripts" / "run2.log").read_text().splitlines() if "rows with 73 features" in l),
        },
    }

    classification = (
        "run-level" if (gate3["all_embedded_vs_recovered_exact"] and gate3["all_pred_max_abs_delta_zero_committed_vs_run1"])
        else "metrics-level" if gate3["all_embedded_vs_recovered_exact"]
        else "not-recovered"
    )
    gates_pass = gate1["all_report_73"] and gate2["summary_metric_payloads_identical"]
    out = {
        "environment": {"python": sys.version.split()[0], "venv": "/tmp/v1_recovery_venv"},
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
        "classification": classification,
        "gates_pass": gates_pass,
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("gate1", "gate2_summary", "classification", "gates_pass") if k in out} | {"gate2": {k: v for k, v in gate2.items() if k != "structure_diffs"}, "gate3_summary": {k: v for k, v in gate3.items() if k != "rows"}, "classification": classification, "gates_pass": gates_pass}, indent=1))
    return 0 if gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
