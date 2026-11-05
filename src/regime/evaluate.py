"""Frozen single-pass REPORT evaluation (POLICY.md §6).

The frozen policy is applied to REPORT origins exactly once. Every
uncertainty is an origin-block bootstrap CI (origins are the independent
unit; folds within an origin are dependent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.model import (
    GUARDRAIL1_MAX_LOSS_PP,
    FrozenPolicy,
    PairSums,
    policy_pair_sums,
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
REL_DECILES = 10


def _pair_index(table: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [table["series_id"].astype(str).to_numpy(), pd.DatetimeIndex(table["forecast_origin"])],
        names=["series_id", "forecast_origin"],
    )


def category_outcome(policy_sums: PairSums, reference: PairSums) -> dict[str, object]:
    """Primary metric inputs: per-category WAPEs and the win count."""
    p = policy_sums.category_wapes()
    r = reference.category_wapes()
    improved = p < r.reindex(p.index)
    return {
        "policy_category_wape": {k: float(v) for k, v in p.items()},
        "reference_category_wape": {k: float(v) for k, v in r.items()},
        "categories_improved": int(improved.sum()),
        "categories_total": int(len(p)),
        "improved_categories": sorted(improved.loc[improved].index.tolist()),
    }


def mean_category_delta_pp(policy_sums: PairSums, reference: PairSums) -> float:
    return 100.0 * (reference.mean_category_wape() - policy_sums.mean_category_wape())


def brier_score(probs: pd.Series, labels: pd.Series) -> float:
    aligned = pd.concat([probs.rename("p"), labels.astype(float).rename("y")], axis=1).dropna()
    return float(np.mean((aligned["p"] - aligned["y"]) ** 2))


def reliability_table(probs: pd.Series, labels: pd.Series, bins: int = REL_DECILES) -> pd.DataFrame:
    """P(theta wins) vs realized theta-win rate, quantile bins."""
    frame = pd.DataFrame({"p": probs, "y": labels.astype(float)}).dropna()
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=bins, labels=False)
    grouped = frame.groupby("bin", sort=True)
    return grouped.agg(
        n=("y", "size"),
        mean_p=("p", "mean"),
        realized_rate=("y", "mean"),
        p_min=("p", "min"),
        p_max=("p", "max"),
    )


def regime_calls(probs: pd.Series, labels: pd.Series, tau: float) -> dict[str, float]:
    """Precision/recall of theta-calls vs realized labels (theta = positive)."""
    frame = pd.DataFrame({"p": probs, "y": labels.astype(bool)}).dropna()
    called = frame["p"] >= tau
    tp = float((called & frame["y"]).sum())
    fp = float((called & ~frame["y"]).sum())
    fn = float((~called & frame["y"]).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "theta_calls": tp + fp,
        "n": float(len(frame)),
    }


def _restrict(sums: PairSums, origins: pd.DatetimeIndex) -> PairSums:
    keep = sums.sums.index.get_level_values("forecast_origin").isin(origins)
    return PairSums(sums=sums.sums.loc[keep])


def bootstrap_report(
    policy: FrozenPolicy,
    report_table: pd.DataFrame,
    naive_sums: PairSums,
    theta_sums: PairSums,
    blend_sums: PairSums,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Origin-block bootstrap for the primary count, mean-category deltas
    vs all three references, and theta-call precision/recall."""
    probs = policy.probabilities(report_table)
    labels = pd.Series(report_table["theta_wins"].astype(bool).to_numpy(), index=probs.index)
    pair_key = _pair_index(report_table)
    origins = pd.DatetimeIndex(sorted(report_table["forecast_origin"].unique()))
    rng = np.random.default_rng(rng_seed)

    counts: list[float] = []
    delta_n: list[float] = []
    delta_t: list[float] = []
    delta_b: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = pd.DatetimeIndex(rng.choice(origins.to_numpy(), size=len(origins), replace=True))
        keep_rows = report_table["forecast_origin"].isin(sample).to_numpy()
        sel = pd.Series(
            np.where(probs.to_numpy()[keep_rows] >= policy.tau, "theta", "naive"),
            index=pair_key[keep_rows],
        )
        sub_n = _restrict(naive_sums, sample)
        sub_t = _restrict(theta_sums, sample)
        sub_b = _restrict(blend_sums, sample)
        pol = policy_pair_sums(sel, sub_n, sub_t)
        counts.append(float(category_outcome(pol, sub_n)["categories_improved"]))
        delta_n.append(mean_category_delta_pp(pol, sub_n))
        delta_t.append(mean_category_delta_pp(pol, sub_t))
        delta_b.append(mean_category_delta_pp(pol, sub_b))
        calls = regime_calls(
            pd.Series(probs.to_numpy()[keep_rows], index=pair_key[keep_rows]),
            pd.Series(labels.to_numpy()[keep_rows], index=pair_key[keep_rows]),
            policy.tau,
        )
        precisions.append(calls["precision"])
        recalls.append(calls["recall"])

    def ci(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, CI_PERCENTILES)
        return {"lo": float(lo), "hi": float(hi)}

    return {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": rng_seed,
        "unit": "forecast_origin (origin-block bootstrap)",
        "categories_improved_ci": ci(counts),
        "delta_vs_naive_pp_ci": ci(delta_n),
        "delta_vs_theta_pp_ci": ci(delta_t),
        "delta_vs_blend_pp_ci": ci(delta_b),
        "precision_ci": ci(precisions),
        "recall_ci": ci(recalls),
    }
