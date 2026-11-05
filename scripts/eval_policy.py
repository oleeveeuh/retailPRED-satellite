"""Frozen single-pass REPORT evaluation (POLICY.md §6).

Loads the frozen artifact from scripts/train_policy.py (model + isotonic +
tau), builds REPORT features from the vintage panel, and applies the frozen
policy to the 2021-01..2022-10 origins exactly once. Writes
results/policy_report.json, the calibration/regime-call tables, and the two
figures (tau sweep on CHOOSE OOF; reliability on CHOOSE OOF + REPORT).
Holdout origins (>= 2023-01) are never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
from regime.causality import assert_origin_window  # noqa: E402
from regime.evaluate import (  # noqa: E402
    bootstrap_report,
    brier_score,
    category_outcome,
    mean_category_delta_pp,
    regime_calls,
    reliability_table,
)
from regime.features import build_feature_table  # noqa: E402
from regime.labels import build_labels, load_fold_data  # noqa: E402
from regime.model import (  # noqa: E402
    blend_pair_sums,
    pair_sums_from_folds,
    policy_pair_sums,
    restrict_sums,
)
from vintageforensics.harness import availability_as_of  # noqa: E402

ARTIFACT_PATH = common.RESULTS / "policy_frozen.joblib"

BLUE = "#2a78d6"      # slot 1: the switched policy
ORANGE = "#eb6834"    # slot 2: pure theta reference
AQUA = "#1baf7a"      # slot 3: fixed blend reference
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE_INK = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def main() -> None:
    artifact = joblib.load(ARTIFACT_PATH)
    policy = artifact["policy"]
    series_categories = artifact["series_categories"]
    global_theta_weight = artifact["global_theta_weight"]
    oof_probs = artifact["oof_probs"]
    oof_labels = artifact["oof_labels"]
    train_provenance = artifact["provenance"]

    boundary = common.HOLDOUT_MIN_ORIGIN
    panel = common.load_panel()
    naive_data, _ = load_fold_data(common.REFERENCE_FOLDS, "seasonal_naive", boundary)
    theta_data, _ = load_fold_data(common.RESULTS / "folds_h5_theta.csv", "theta_method", boundary)
    label_table = build_labels(naive_data, theta_data)

    full_table = build_feature_table(panel, naive_data.origins, naive_data, theta_data,
                                     availability_as_of)
    choose_end = common.CHOOSE_END
    report = full_table.loc[
        (full_table["forecast_origin"] >= choose_end)
        & (full_table["forecast_origin"] < boundary)
    ].copy()
    report = report.set_index(["series_id", "forecast_origin"]).join(
        label_table.labels.rename("theta_wins")
    ).join(label_table.naive_wape).join(label_table.theta_wape).reset_index()
    report_origins = pd.DatetimeIndex(sorted(report["forecast_origin"].unique()))
    assert_origin_window(report_origins, boundary)
    if report_origins.min() < pd.Timestamp(choose_end):
        raise AssertionError("REPORT window leaks into CHOOSE")
    if report["error_features_pending"].any() or report["theta_wins"].isna().any():
        raise AssertionError("REPORT rows incomplete (features or labels)")
    pair_key = pd.MultiIndex.from_arrays(
        [report["series_id"].astype(str).to_numpy(), pd.DatetimeIndex(report["forecast_origin"])],
        names=["series_id", "forecast_origin"],
    )

    probs = policy.probabilities(report)
    probs.index = pair_key  # positional: probabilities() preserves report row order
    selection = (probs >= policy.tau).map({True: "theta", False: "naive"})
    labels = pd.Series(report["theta_wins"].astype(bool).to_numpy(), index=pair_key)

    naive_all = pair_sums_from_folds(naive_data.folds)
    theta_all = pair_sums_from_folds(theta_data.folds)
    blend_all = blend_pair_sums(naive_data.folds, theta_data.folds, global_theta_weight)
    naive_rep = restrict_sums(naive_all, report_origins)
    theta_rep = restrict_sums(theta_all, report_origins)
    blend_rep = restrict_sums(blend_all, report_origins)
    policy_rep = policy_pair_sums(selection, naive_rep, theta_rep)

    vs_naive = category_outcome(policy_rep, naive_rep)
    mean_cat_policy = policy_rep.mean_category_wape()
    mean_cat_naive = naive_rep.mean_category_wape()
    mean_cat_theta = theta_rep.mean_category_wape()
    mean_cat_blend = blend_rep.mean_category_wape()
    delta_naive_pp = mean_category_delta_pp(policy_rep, naive_rep)
    delta_theta_pp = mean_category_delta_pp(policy_rep, theta_rep)
    delta_blend_pp = mean_category_delta_pp(policy_rep, blend_rep)
    beats_everything = bool(
        mean_cat_policy < mean_cat_naive
        and mean_cat_policy < mean_cat_theta
        and mean_cat_policy < mean_cat_blend
    )

    calls_rep = regime_calls(probs, labels, policy.tau)
    brier_rep = brier_score(probs, labels)
    rel_rep = reliability_table(probs, labels)
    calls_oof = regime_calls(oof_probs, oof_labels, policy.tau)
    rel_oof = reliability_table(oof_probs, oof_labels)

    boots = bootstrap_report(policy, report, naive_all, theta_all, blend_all)

    def rel_records(frame: pd.DataFrame) -> list[dict]:
        return [
            {key: (int(v) if key == "n" else float(v)) for key, v in row.items()}
            for row in frame.reset_index().to_dict("records")
        ]

    payload = {
        "provenance": common.provenance({
            "frozen_artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
            "train_provenance_generated_at": train_provenance["generated_at"],
            "n_pairs_report": int(len(report)),
            "n_pairs_choose_oof": int(len(oof_probs)),
            "sample_size_note": (
                f"n={len(report)} REPORT pairs (9 origins x 11 categories); single frozen "
                "evaluation per POLICY.md §2/§6; uncertainties are origin-block bootstrap CIs"
            ),
            "report_origins": [str(o.date()) for o in report_origins],
            "model": policy.kind,
            "tau": float(policy.tau),
        }),
        "frozen_policy": {
            "model": policy.kind,
            "tau": float(policy.tau),
            "oof_brier_choose": float(policy.oof_brier),
            "choose_policy_mean_cat_wape": float(policy.choose_policy_wape),
        },
        "primary_categories_beating_naive": {
            "count": vs_naive["categories_improved"],
            "total": vs_naive["categories_total"],
            "improved_categories": vs_naive["improved_categories"],
            "policy_category_wape": vs_naive["policy_category_wape"],
            "naive_category_wape": vs_naive["reference_category_wape"],
        },
        "secondary_mean_category_wape": {
            "policy": mean_cat_policy,
            "naive": mean_cat_naive,
            "theta": mean_cat_theta,
            "blend_050": mean_cat_blend,
            "delta_vs_naive_pp": delta_naive_pp,
            "delta_vs_theta_pp": delta_theta_pp,
            "delta_vs_blend_pp": delta_blend_pp,
        },
        "guardrail2_beats_everything": {
            "beats_naive": bool(mean_cat_policy < mean_cat_naive),
            "beats_theta": bool(mean_cat_policy < mean_cat_theta),
            "beats_blend": bool(mean_cat_policy < mean_cat_blend),
            "passes": beats_everything,
        },
        "calibration": {
            "brier_report_frozen": brier_rep,
            "brier_choose_oof": float(policy.oof_brier),
            "reliability_report_deciles": rel_records(rel_rep),
            "reliability_choose_oof_deciles_isotonic_in_sample": rel_records(rel_oof),
        },
        "regime_calls": {
            "report_frozen": {k: float(v) for k, v in calls_rep.items()},
            "choose_oof": {k: float(v) for k, v in calls_oof.items()},
            **boots,
        },
        "theta_call_rate_report": float((selection == "theta").mean()),
    }
    common.write_result("policy_report", payload)

    write_tau_sweep_figure(policy, mean_cat_policy)
    write_reliability_figure(rel_oof, rel_rep, float(policy.oof_brier), brier_rep)

    print(f"policy {policy.kind} tau {policy.tau:.2f} | REPORT mean-category WAPE "
          f"{mean_cat_policy:.4f} vs naive {mean_cat_naive:.4f} theta {mean_cat_theta:.4f} "
          f"blend {mean_cat_blend:.4f}")
    print(f"primary: {vs_naive['categories_improved']}/{vs_naive['categories_total']} "
          f"categories beat naive | guardrail 2 (beat everything): {beats_everything}")
    print(f"Brier REPORT {brier_rep:.4f} (CHOOSE OOF {policy.oof_brier:.4f}) | "
          f"theta-call precision {calls_rep['precision']:.3f} recall {calls_rep['recall']:.3f}")


def write_tau_sweep_figure(policy, report_policy_wape: float) -> None:
    sweep = policy.sweep
    refs = sweep.attrs.get("reference_mean_cat", {})
    tau = float(policy.tau)
    chosen_row = sweep.loc[np.isclose(sweep["tau"], tau)].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sweep["tau"], 100 * sweep["policy_mean_cat_wape"], color=BLUE, lw=2,
            label="switched policy (CHOOSE OOF)", zorder=4)
    ax.axhline(100 * refs.get("naive", np.nan), color=BASELINE_INK, lw=1.4, ls=(0, (4, 3)),
               zorder=2)
    ax.axhline(100 * refs.get("theta", np.nan), color=ORANGE, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axhline(100 * refs.get("blend", np.nan), color=AQUA, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axvline(tau, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
    ax.scatter([tau], [100 * chosen_row["policy_mean_cat_wape"]], s=64, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax.scatter([tau], [100 * report_policy_wape], s=72, marker="D", color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=6)

    floor = min(100 * sweep["policy_mean_cat_wape"].min(), 100 * report_policy_wape)
    ax.set_ylim(floor * 0.90, 100 * sweep["policy_mean_cat_wape"].max() * 1.02)
    label_x = 1.01
    for value, color, name in (
        (refs.get("naive"), BASELINE_INK, "naive"),
        (refs.get("theta"), ORANGE, "pure theta"),
        (refs.get("blend"), AQUA, "fixed blend 0.5"),
    ):
        ax.annotate(f"{name} {100 * value:.2f}", xy=(label_x, 100 * value),
                    xycoords=("axes fraction", "data"), fontsize=8.5, color=color,
                    va="center", annotation_clip=False)
    ax.annotate(f"frozen tau = {tau:.2f}", xy=(tau, 100 * chosen_row["policy_mean_cat_wape"]),
                xytext=(-12, 16), textcoords="offset points", fontsize=8.5, color=MUTED,
                ha="right")
    ax.annotate("REPORT, evaluated once", xy=(tau, 100 * report_policy_wape),
                xytext=(10, -4), textcoords="offset points", fontsize=8.5, color=INK,
                va="top")
    ax.set_xlabel("tau: use theta when P(theta wins) >= tau")
    ax.set_ylabel("mean-category WAPE (%)")
    ax.set_title("Policy tau sweep — CHOOSE origins (out-of-fold); diamond = frozen REPORT point",
                 fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "policy_tau_sweep.png", bbox_inches="tight")
    plt.close(fig)


def write_reliability_figure(rel_oof: pd.DataFrame, rel_rep: pd.DataFrame,
                             brier_oof: float, brier_rep: float) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = (0, 1)
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("perfect calibration", xy=(0.36, 0.41), fontsize=8.5, color=MUTED, rotation=38)

    ax.plot(rel_oof["mean_p"], rel_oof["realized_rate"], color=BLUE, lw=1.6, zorder=3)
    ax.scatter(rel_oof["mean_p"], rel_oof["realized_rate"], s=52, color=BLUE,
               edgecolor=SURFACE, linewidth=1.2, zorder=4,
               label=f"CHOOSE OOF (n={int(rel_oof['n'].sum())}, Brier {brier_oof:.3f})")
    ax.scatter(rel_rep["mean_p"], rel_rep["realized_rate"], s=64, marker="D", color=ORANGE,
               edgecolor=SURFACE, linewidth=1.2, zorder=5,
               label=f"REPORT frozen (n={int(rel_rep['n'].sum())}, Brier {brier_rep:.3f})")
    for i, (_, row) in enumerate(rel_rep.iterrows()):
        ax.annotate(f"n={int(row['n'])}", (row["mean_p"], row["realized_rate"]),
                    xytext=(8 if i % 2 == 0 else 8, 6 if i % 2 == 0 else -10),
                    textcoords="offset points", fontsize=7.5, color=INK_2)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean P(theta wins) in decile bin")
    ax.set_ylabel("realized theta-win rate")
    ax.set_title("Reliability of P(theta wins) — deciles of scored pairs",
                 fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "policy_reliability.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
