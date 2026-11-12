#!/usr/bin/env python3
"""v1 recovery — pickle audit (Phase 2 step: loads attempted BEFORE any training).

Loads every committed reference pickle from /tmp/v1_recovery_reference/,
verifies SHA-256 against the predeclared inventory, and records per pickle:
loaded / unpickling error, model class, embedded metrics, feature count,
training_time, and any library-version mismatch warnings. Nothing is
substituted: a pickle that fails to unpickle is reported with its exact
error. Output: JSON transcript on stdout (also written to the path given
as argv[1]).

Run with the recovery venv's interpreter:
    /tmp/v1_recovery_venv/bin/python scripts/v1_recovery_pickle_audit.py <out.json>
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

REFERENCE = Path("/tmp/v1_recovery_reference")
INVENTORY = REFERENCE / "input_shas.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_model_data(name: str, md) -> dict:
    """Extract evidence fields from a model_data dict without assuming its schema."""
    out = {"keys": sorted(md.keys()) if isinstance(md, dict) else f"NOT A DICT: {type(md).__name__}"}
    if not isinstance(md, dict):
        return out
    model = md.get("model")
    if model is not None:
        out["model_class"] = type(model).__name__
        mod = type(model).__module__
        out["model_module"] = mod
        # library version recorded inside the pickle, when available
        try:
            import sklearn
            out["runtime_sklearn"] = sklearn.__version__
        except Exception:
            pass
    if isinstance(md.get("metrics"), dict):
        out["embedded_metrics"] = md["metrics"]
    if "n_features" in md:
        out["embedded_n_features"] = md["n_features"]
    if isinstance(md.get("features"), list):
        out["embedded_features_len"] = len(md["features"])
    for k in ("training_time", "model_name", "is_trained"):
        if k in md:
            out[k] = str(md[k])
    return out


def main() -> int:
    import joblib

    out_path = Path(sys.argv[1])
    inventory = json.loads(INVENTORY.read_text())
    records = []
    n_loaded = n_failed = 0
    for group in ("pickles_backend_ml_models", "pickles_backend_training_outputs_models"):
        for rel, expected_sha in sorted(inventory[group].items()):
            # reference layout mirrors the two source roots
            p = REFERENCE / rel.split("backend/", 1)[1] if group.endswith("ml_models") else None
            if group == "pickles_backend_ml_models":
                p = REFERENCE / "backend_ml_models" / Path(rel).name
            else:
                cat = Path(rel).parent.name
                p = REFERENCE / "backend_training_outputs_models" / cat / Path(rel).name
            rec = {"pickle": rel, "sha256_matches_inventory": sha256(p) == expected_sha}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    md = joblib.load(p)
                    rec["status"] = "loaded"
                    rec.update(describe_model_data(rel, md))
                    n_loaded += 1
                except Exception as e:  # noqa: BLE001 — the error text IS the finding
                    rec["status"] = "unpickle_failed"
                    rec["error"] = f"{type(e).__name__}: {e}"
                    n_failed += 1
            rec["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
            records.append(rec)

    result = {
        "environment": {"python": sys.version.split()[0]},
        "n_loaded": n_loaded,
        "n_failed": n_failed,
        "all_shas_match_inventory": all(r["sha256_matches_inventory"] for r in records),
        "records": records,
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
