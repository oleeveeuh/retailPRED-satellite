#!/usr/bin/env python3
"""H9 final scorecard: the one surviving configuration, scored once (§6, §8).

Reporting-only in the REGATE sense: this script runs AFTER the ladder
concludes, computes the report window for the first time, and applies the
frozen REGATE v2 mirror (owner-confirmed win rule) — it has no selection
input and changes no model. Every number it releases is gated:

Verification gates (STOP on mismatch, §7 pattern):
  1. h9_rung_anchor.json must record verification_gate_7_1.pass=true (the
     anchor reproduced the H7 survivor folds at 1e-6), and the surviving
     rung's own gate record (alignment 7.5) must be non-degenerate when active.
  2. Candidate/naive/theta fold keys must be identical (paired metrics).
  3. Every origin < HOLDOUT_MIN_ORIGIN and equal to the frozen exploration
     grid; as-of rule verified against stored origin_as_of columns.
  4. Naive and theta full-exploration mean-category WAPE must reproduce the
     frozen Phase-2 record (results/cand_h5_theta.json windows.full) at
     1e-9 before any delta is computed.

Windows: choose (already known), report (computed here for the first time),
full exploration. Actuals: final-vintage (governing) and first-release
(descriptive companion, never a pass/fail input). Metrics: pooled WAPE,
mean-category WAPE, MASE (point-in-time seasonal scale), relMAE vs naive,
signed bias, per-horizon, categories-beating counts vs naive AND theta,
origin-block bootstrap CIs (B=2000, seed 42) vs both benchmarks, and the
C1/C2/C3 REGATE-mirror verdict per window — reported, never a promotion
claim (promotion = future origins under REGATE_DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    HOLDOUT_MIN_ORIGIN,
    RESULTS,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
PRACTICAL_FLOOR_PP = 0.25
POOLED_GUARD_PP = -0.25
CRITICAL_CATEGORY = "retail_total"
CATEGORY_BOUND_PP = 2.0
CRITICAL_BOUND_PP = 1.0

FROZEN_PHASE2_RECORD = RESULTS / "cand_h5_theta.json"
ANCHOR_RECORD = RESULTS / "h9_rung_anchor.json"
THETA_FOLDS = RESULTS / "folds_h5_theta.csv"


def _stop(message: str) -> None:
    sys.exit(f"STOP: {message} Investigate before releasing this scorecard.")


def fold_keys(frame: pd.DataFrame) -> frozenset:
    return frozenset(
        zip(frame["series_id"], frame["forecast_origin"], frame["target_time"], frame["horizon"])
    )


def first_release_actuals(panel: pd.DataFrame) -> pd.DataFrame:
    ordered = panel.sort_values(["vintage_time", "observation_time"])
    first = ordered.drop_duplicates(["series_id", "observation_time"], keep="first")
    return first[["series_id", "observation_time", "value"]].rename(
        columns={"value": "fr_actual"}
    )


def mase_scales(panel: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Point-in-time seasonal-naive scale (Hyndman-Koehler), ported from
    scripts/holdout_scorecard.py; the as-of rule is verified first."""
    rows = []
    for frame in frames:
        computed = (
            frame["forecast_origin"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)
        ).dt.tz_localize("UTC")
        stored = pd.to_datetime(frame["origin_as_of"], utc=True)
        if not (computed == stored).all():
            _stop("as-of rule (origin month-end + 45 days) does not match stored origin_as_of.")
        rows.append(frame[["series_id", "forecast_origin", "training_cutoff"]].drop_duplicates())
    keys = pd.concat(rows, ignore_index=True).drop_duplicates(["series_id", "forecast_origin"])
    scales = []
    for series_id, origin, cutoff in keys.itertuples(index=False):
        as_of = origin + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)
        obs = fast_observations_as_of(panel, pd.Timestamp(as_of).tz_localize("UTC"))
        window = obs.loc[
            (obs["series_id"] == series_id) & (obs["observation_time"] <= cutoff)
        ].sort_values("observation_time")
        values = pd.Series(
            window["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(window["observation_time"]),
        )
        lagged = values.reindex(values.index - pd.DateOffset(months=12))
        diffs = (values - lagged.to_numpy()).abs().dropna()
        scale = float(diffs.mean()) if len(diffs) else float("nan")
        if not np.isfinite(scale) or scale <= 0:
            _stop(f"degenerate MASE scale for {series_id} at {origin.date()}.")
        scales.append({"series_id": series_id, "forecast_origin": origin, "mase_scale": scale})
    return pd.DataFrame(scales)


def prepare(frame: pd.DataFrame, scales: pd.DataFrame, first_release: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(scales, on=["series_id", "forecast_origin"], how="left", validate="many_to_one")
    out = out.merge(
        first_release,
        left_on=["series_id", "target_time"],
        right_on=["series_id", "observation_time"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["observation_time"])
    if out["mase_scale"].isna().any() or out["fr_actual"].isna().any():
        _stop("MASE scale or first-release actual coverage incomplete after merge.")
    return out


def _wape(actual: pd.Series, prediction: pd.Series) -> float:
    return float((prediction - actual).abs().sum() / actual.abs().sum())


def model_stats(frame: pd.DataFrame, actual_col: str) -> dict:
    act, pred = frame[actual_col], frame["prediction"]
    err = pred - act
    per_series = {}
    for series_id, group in frame.groupby("series_id", sort=True):
        per_series[str(series_id)] = {
            "wape": _wape(group[actual_col], group["prediction"]),
            "mae": float((group["prediction"] - group[actual_col]).abs().mean()),
            "mase": float(((group["prediction"] - group[actual_col]).abs() / group["mase_scale"]).mean()),
            "mpe_pp": float((((group["prediction"] - group[actual_col]) / group[actual_col]).mean()) * 100),
        }
    by_horizon = {}
    for horizon, group in frame.groupby("horizon", sort=True):
        by_horizon[str(int(horizon))] = {
            "mean_category": float(
                pd.Series(
                    {str(s): _wape(g[actual_col], g["prediction"]) for s, g in group.groupby("series_id")}
                ).mean()
            ),
            "pooled": _wape(group[actual_col], group["prediction"]),
        }
    return {
        "fold_rows": int(len(frame)),
        "pooled_wape": _wape(act, pred),
        "mean_category_wape": float(pd.Series({s: b["wape"] for s, b in per_series.items()}).mean()),
        "mase_macro": float(pd.Series({s: b["mase"] for s, b in per_series.items()}).mean()),
        "pooled_relative_bias_pp": float(err.sum() / act.sum() * 100),
        "by_series": per_series,
        "by_horizon": by_horizon,
    }


def rel_mae_vs_naive(model: dict, naive: dict) -> float:
    return float(
        pd.Series(
            {
                s: model["by_series"][s]["mae"] / naive["by_series"][s]["mae"]
                for s in sorted(model["by_series"])
            }
        ).mean()
    )


def _pivot_sums(frame: pd.DataFrame, actual_col: str) -> tuple[np.ndarray, np.ndarray]:
    work = frame.assign(_abs_err=(frame["prediction"] - frame[actual_col]).abs())
    se = work.pivot_table(index="forecast_origin", columns="series_id", values="_abs_err", aggfunc="sum")
    sy = work.pivot_table(index="forecast_origin", columns="series_id", values=actual_col, aggfunc="sum")
    return se.to_numpy(), sy.to_numpy()


def bootstrap_delta(candidate: pd.DataFrame, benchmark: pd.DataFrame, actual_col: str) -> dict:
    """Origin-block bootstrap (whole origins, multiplicity, B=2000, seed 42),
    ported from scripts/holdout_scorecard.py; positive delta = candidate better."""
    se_c, sy_c = _pivot_sums(candidate, actual_col)
    se_b, sy_b = _pivot_sums(benchmark, actual_col)
    n = len(se_c)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    wc = se_c.sum(axis=0) / sy_c.sum(axis=0)
    wb = se_b.sum(axis=0) / sy_b.sum(axis=0)
    point = {
        "mean_category_delta_pp": float(100 * (wb.mean() - wc.mean())),
        "pooled_delta_pp": float(100 * (se_b.sum() / sy_b.sum() - se_c.sum() / sy_c.sum())),
        "categories_improved": int((wc < wb).sum()),
    }
    deltas_mc, deltas_pooled, counts = [], [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        wc_b = se_c[idx].sum(axis=0) / sy_c[idx].sum(axis=0)
        wb_b = se_b[idx].sum(axis=0) / sy_b[idx].sum(axis=0)
        deltas_mc.append(100 * (wb_b.mean() - wc_b.mean()))
        deltas_pooled.append(100 * (se_b[idx].sum() / sy_b[idx].sum() - se_c[idx].sum() / sy_c[idx].sum()))
        counts.append(int((wc_b < wb_b).sum()))

    def ci(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        lo, hi = np.percentile(arr, CI_PERCENTILES)
        return {"lo": float(lo), "hi": float(hi)}

    return {
        "n_origins": int(n),
        "point": point,
        "mean_category_delta_pp_ci": ci(deltas_mc),
        "pooled_delta_pp_ci": ci(deltas_pooled),
        "categories_improved_ci": ci([float(c) for c in counts]),
    }


def regate_mirror(stats: dict, naive_stats: dict, theta_stats: dict,
                  boot_naive: dict, boot_theta: dict, actual_col: str) -> dict:
    """C1/C2/C3 of REGATE_DESIGN.md §3, evaluated on this window. Reported,
    never a promotion claim (§1 of H7_FEATURE_LADDER.md)."""
    ci_naive = boot_naive["mean_category_delta_pp_ci"]
    ci_theta = boot_theta["mean_category_delta_pp_ci"]
    c1 = {
        "ci_lo_pp": ci_naive["lo"],
        "ci_wholly_positive": bool(ci_naive["lo"] > 0),
        "point_pp": boot_naive["point"]["mean_category_delta_pp"],
        "meets_floor": bool(boot_naive["point"]["mean_category_delta_pp"] >= PRACTICAL_FLOOR_PP),
        "pooled_point_pp": boot_naive["point"]["pooled_delta_pp"],
        "pooled_guard_pass": bool(boot_naive["point"]["pooled_delta_pp"] >= POOLED_GUARD_PP),
    }
    c1["pass"] = bool(c1["ci_wholly_positive"] and c1["meets_floor"] and c1["pooled_guard_pass"])
    regress_naive = {
        s: 100 * (stats["by_series"][s]["wape"] - naive_stats["by_series"][s]["wape"])
        for s in stats["by_series"]
    }
    regress_theta = {
        s: 100 * (stats["by_series"][s]["wape"] - theta_stats["by_series"][s]["wape"])
        for s in stats["by_series"]
    }
    c2 = {
        "worst_category_regression_pp": max(regress_naive.values()),
        "worst_category": max(regress_naive, key=regress_naive.get),
        "critical_category_regression_pp": regress_naive[CRITICAL_CATEGORY],
        "pass": bool(
            max(regress_naive.values()) <= CATEGORY_BOUND_PP
            and regress_naive[CRITICAL_CATEGORY] <= CRITICAL_BOUND_PP
        ),
    }
    c3 = {
        "ci_hi_pp": ci_theta["hi"],
        "ci_not_wholly_negative": bool(ci_theta["hi"] >= 0),
        "worst_category_regression_pp": max(regress_theta.values()),
        "worst_category_vs_theta": max(regress_theta, key=regress_theta.get),
    }
    c3["pass"] = bool(c3["ci_not_wholly_negative"] and c3["worst_category_regression_pp"] <= CATEGORY_BOUND_PP)
    return {
        "actual_set": actual_col,
        "C1_aggregate_vs_naive": c1,
        "C2_no_critical_harm_vs_naive": c2,
        "C3_frontier_vs_theta": c3,
        "all_pass": bool(c1["pass"] and c2["pass"] and c3["pass"]),
        "note": "exploration-side selection verdict; promotion requires future origins (REGATE_DESIGN)",
    }


def category_counts(stats: dict, benchmark: dict) -> int:
    return sum(1 for s in stats["by_series"] if stats["by_series"][s]["wape"] < benchmark["by_series"][s]["wape"])


def window_block(label: str, prepared: dict, actual_col: str) -> dict:
    stats = {name: model_stats(frame, actual_col) for name, frame in prepared.items()}
    boot_naive = bootstrap_delta(prepared["candidate"], prepared["naive"], actual_col)
    boot_theta = bootstrap_delta(prepared["candidate"], prepared["theta"], actual_col)
    block = {
        "n_origins": int(prepared["candidate"]["forecast_origin"].nunique()),
        "stats": stats,
        "rel_mae_vs_naive": rel_mae_vs_naive(stats["candidate"], stats["naive"]),
        "categories_beating_naive": category_counts(stats["candidate"], stats["naive"]),
        "categories_beating_theta": category_counts(stats["candidate"], stats["theta"]),
        "bootstrap_vs_naive": boot_naive,
        "bootstrap_vs_theta": boot_theta,
    }
    if actual_col == "actual":  # the mirror applies to the governing truth only
        block["regate_mirror"] = regate_mirror(
            stats["candidate"], stats["naive"], stats["theta"], boot_naive, boot_theta, actual_col
        )
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-stem", required=True, help="e.g. direct_growth")
    parser.add_argument("--label", default=None, help="candidate label for the report")
    args = parser.parse_args()

    anchor = json.loads(ANCHOR_RECORD.read_text())
    if not (anchor.get("verification_gate_7_1") or {}).get("pass"):
        _stop("h9_rung_anchor.json does not record verification_gate_7_1.pass=true.")
    rung_record = RESULTS / f"h9_rung_{args.folds_stem}.json"
    if rung_record.exists():
        rung = json.loads(rung_record.read_text())
        gate_7_5 = rung.get("alignment_gate_7_5")
        if gate_7_5 is not None and gate_7_5["diff_count"] <= 0:
            _stop("surviving rung's alignment gate records a degenerate mechanism (7.5).")

    frozen = json.loads(FROZEN_PHASE2_RECORD.read_text())
    frozen_naive = frozen["windows"]["full"]["naive"]["mean_category_wape"]
    frozen_theta = frozen["windows"]["full"]["candidate"]["mean_category_wape"]

    panel = load_panel()
    candidate = pd.read_csv(RESULTS / f"h9_folds_{args.folds_stem}.csv",
                            parse_dates=["forecast_origin", "target_time"])
    naive = naive_reference_for(sorted(pd.DatetimeIndex(candidate["forecast_origin"].unique())))
    theta = pd.read_csv(THETA_FOLDS, parse_dates=["forecast_origin", "target_time"])
    theta = theta.loc[theta["forecast_origin"] < HOLDOUT_MIN_ORIGIN]

    for name, frame in (("candidate", candidate), ("naive", naive), ("theta", theta)):
        if frame["forecast_origin"].max() >= HOLDOUT_MIN_ORIGIN:
            _stop(f"{name} folds contain holdout origins.")
    if not fold_keys(candidate) == fold_keys(naive) == fold_keys(theta):
        _stop("candidate/naive/theta fold keys differ; paired metrics require identical folds.")
    grid = pd.DatetimeIndex(sorted(candidate["forecast_origin"].unique()))
    if not grid.equals(pd.DatetimeIndex(sorted(theta["forecast_origin"].unique()))):
        _stop("origin grid differs from the frozen exploration grid.")

    def mean_category_wape_only(frame: pd.DataFrame) -> float:
        return float(
            pd.Series(
                {
                    str(series): _wape(group["actual"], group["prediction"])
                    for series, group in frame.groupby("series_id", sort=True)
                }
            ).mean()
        )

    naive_check = mean_category_wape_only(naive)
    theta_check = mean_category_wape_only(theta)
    if abs(naive_check - frozen_naive) > 1e-9:
        _stop(f"naive reproduction failed: {naive_check} != {frozen_naive}")
    if abs(theta_check - frozen_theta) > 1e-9:
        _stop(f"theta reproduction failed: {theta_check} != {frozen_theta}")

    # fold keys are already verified identical; fix a common row order so
    # boolean window masks apply positionally to all three frames
    sort_keys = ["series_id", "forecast_origin", "target_time", "horizon"]
    candidate = candidate.sort_values(sort_keys, ignore_index=True)
    naive = naive.sort_values(sort_keys, ignore_index=True)
    theta = theta.sort_values(sort_keys, ignore_index=True)

    scales = mase_scales(panel, [candidate, naive, theta])
    first_release = first_release_actuals(panel)
    prepared = {name: prepare(frame, scales, first_release) for name, frame in
                (("candidate", candidate), ("naive", naive), ("theta", theta))}

    choose_mask = (
        pd.to_datetime(prepared["candidate"]["forecast_origin"]) < pd.Timestamp("2021-01-01")
    ).to_numpy()
    windows = {}
    for label, mask in (
        ("choose", choose_mask),
        ("report", ~choose_mask),
        ("full_exploration", np.ones(len(choose_mask), dtype=bool)),
    ):
        selected = {name: frame.loc[mask] for name, frame in prepared.items()}
        if any(frame.empty for frame in selected.values()):
            _stop(f"window {label} selected no rows.")
        windows[label] = {
            "final_vintage": window_block(label, selected, "actual"),
            "first_release_descriptive": window_block(label, selected, "fr_actual"),
        }

    payload = {
        "provenance": provenance({"purpose": f"H9 final scorecard for {args.folds_stem}"}),
        "candidate": args.label or args.folds_stem,
        "verification": {
            "anchor_reproduction_gate_7_1": anchor.get("verification_gate_7_1"),
            "frozen_naive_mean_category_wape": frozen_naive,
            "frozen_theta_mean_category_wape": frozen_theta,
            "n_origins": int(len(grid)),
            "origin_first": grid.min().date().isoformat(),
            "origin_last": grid.max().date().isoformat(),
        },
        "windows": windows,
    }
    out_json = RESULTS / "h9_scorecard.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        "# H9 scorecard — final configuration",
        "",
        f"Candidate: **{args.label or args.folds_stem}** | {len(grid)} exploration origins "
        f"({grid.min().date()}..{grid.max().date()}) | verification gates passed "
        "(h=1 equivalence, fold-key identity, naive/theta frozen reproduction).",
        "",
        "All deltas paired, positive = candidate better. Actuals: final-vintage "
        "(governing); first-release re-scoring is the descriptive companion.",
        "",
    ]
    for label in ("choose", "report", "full_exploration"):
        for actual_key, actual_name in (("final_vintage", "final-vintage actuals"),
                                        ("first_release_descriptive", "first-release actuals (descriptive)")):
            block = windows[label][actual_key]
            stats = block["stats"]
            boot_n = block["bootstrap_vs_naive"]
            boot_t = block["bootstrap_vs_theta"]
            lines += [
                f"## {label} — {actual_name} ({block['n_origins']} origins)",
                "",
                "| model | pooled WAPE | mean-cat WAPE | MASE | bias | h=1 | h=3 | h=6 | h=12 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for name in ("naive", "theta", "candidate"):
                s = stats[name]
                horizons = s["by_horizon"]
                lines.append(
                    f"| {name} | {100 * s['pooled_wape']:.2f}% | {100 * s['mean_category_wape']:.2f}% | "
                    f"{s['mase_macro']:.3f} | {s['pooled_relative_bias_pp']:+.2f}pp | "
                    f"{100 * horizons['1']['mean_category']:.2f} | {100 * horizons['3']['mean_category']:.2f} | "
                    f"{100 * horizons['6']['mean_category']:.2f} | {100 * horizons['12']['mean_category']:.2f} |"
                )
            lines += [
                "",
                f"- relMAE (candidate vs naive): {block['rel_mae_vs_naive']:.3f} | "
                f"categories beating naive: {block['categories_beating_naive']}/11 | "
                f"beating theta: {block['categories_beating_theta']}/11",
                f"- vs naive mean-cat: {boot_n['point']['mean_category_delta_pp']:+.2f}pp "
                f"(CI {boot_n['mean_category_delta_pp_ci']['lo']:+.2f}..{boot_n['mean_category_delta_pp_ci']['hi']:+.2f}) | "
                f"pooled {boot_n['point']['pooled_delta_pp']:+.2f}pp "
                f"(CI {boot_n['pooled_delta_pp_ci']['lo']:+.2f}..{boot_n['pooled_delta_pp_ci']['hi']:+.2f})",
                f"- vs theta mean-cat: {boot_t['point']['mean_category_delta_pp']:+.2f}pp "
                f"(CI {boot_t['mean_category_delta_pp_ci']['lo']:+.2f}..{boot_t['mean_category_delta_pp_ci']['hi']:+.2f}) | "
                f"pooled {boot_t['point']['pooled_delta_pp']:+.2f}pp "
                f"(CI {boot_t['pooled_delta_pp_ci']['lo']:+.2f}..{boot_t['pooled_delta_pp_ci']['hi']:+.2f})",
            ]
            if actual_key == "final_vintage":
                mirror = block["regate_mirror"]
                c1, c2, c3 = mirror["C1_aggregate_vs_naive"], mirror["C2_no_critical_harm_vs_naive"], mirror["C3_frontier_vs_theta"]
                lines += [
                    "",
                    f"REGATE v2 mirror: C1 {'PASS' if c1['pass'] else 'FAIL'} "
                    f"(CI {c1['ci_lo_pp']:+.2f}.., floor {c1['point_pp']:+.2f}pp, pooled guard {c1['pooled_point_pp']:+.2f}pp) | "
                    f"C2 {'PASS' if c2['pass'] else 'FAIL'} "
                    f"(worst {c2['worst_category']} {c2['worst_category_regression_pp']:+.2f}pp, "
                    f"{CRITICAL_CATEGORY} {c2['critical_category_regression_pp']:+.2f}pp) | "
                    f"C3 {'PASS' if c3['pass'] else 'FAIL'} "
                    f"(CI ..{c3['ci_hi_pp']:+.2f}, worst vs theta {c3['worst_category_vs_theta']} "
                    f"{c3['worst_category_regression_pp']:+.2f}pp) → "
                    f"**{'ALL PASS (selection verdict only — not a promotion)' if mirror['all_pass'] else 'FAIL'}**",
                ]
            lines.append("")
    lines += [
        "Coarse-CI note: 9 report / 24 full-exploration origin blocks — intervals are "
        "wide by construction and reported as such (scorecard precedent).",
        "",
        "This scorecard is an exploration-side selection result. It promotes nothing; "
        "the only promotion path is the future-origin re-gate (REGATE_DESIGN.md).",
        "",
    ]
    out_md = RESULTS / "h9_scorecard.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_json.name} and {out_md.name}")

    report = windows["report"]["final_vintage"]
    mirror = report["regate_mirror"]
    print(f"report window mean-cat: candidate {100 * report['stats']['candidate']['mean_category_wape']:.2f}% | "
          f"naive {100 * report['stats']['naive']['mean_category_wape']:.2f}% | "
          f"theta {100 * report['stats']['theta']['mean_category_wape']:.2f}%")
    print(f"REGATE v2 mirror (report): {'ALL PASS' if mirror['all_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
