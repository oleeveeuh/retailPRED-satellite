"""Policy model, calibration, and the tau decision rule (POLICY.md §5).

Two capped model forms (logistic baseline, LightGBM challenger), out-of-fold
scores via leave-one-origin-out on CHOOSE, isotonic calibration on those
OOF scores only, and a full tau sweep whose optimum is chosen under the
bounded-downside guardrail. No hyperparameter search exists anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from regime.features import NUMERIC_FEATURES

MODEL_KINDS = ("logistic", "lightgbm")
GUARDRAIL1_MAX_LOSS_PP = 0.5
TAU_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 4)
RANDOM_STATE = 42

LGBM_PARAMS = {
    "n_estimators": 100,
    "num_leaves": 8,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def design_matrix(table: pd.DataFrame, series_categories: list[str]) -> pd.DataFrame:
    """Feature columns with series identity as a fixed categorical."""
    x = table.loc[:, ["series_id", *NUMERIC_FEATURES]].copy()
    x["series_id"] = pd.Categorical(x["series_id"], categories=series_categories)
    if x[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("design matrix contains NaN features; POLICY.md forbids imputation")
    return x


def make_model(kind: str) -> object:
    if kind == "logistic":
        prep = ColumnTransformer(
            [
                ("num", StandardScaler(), list(NUMERIC_FEATURES)),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    ["series_id"],
                ),
            ]
        )
        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        )
        return Pipeline([("prep", prep), ("clf", clf)])
    if kind == "lightgbm":
        return LGBMClassifier(**LGBM_PARAMS)
    raise ValueError(f"unknown model kind {kind!r}")


def fit_model(kind: str, x: pd.DataFrame, y: pd.Series) -> object:
    model = make_model(kind)
    if kind == "lightgbm":
        model.fit(x, y, categorical_feature=["series_id"])
    else:
        model.fit(x, y)
    return model


def positive_probability(model: object, x: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "lightgbm":
        return model.predict_proba(x)[:, 1]
    return model.predict_proba(x)[:, 1]


def oof_scores(kind: str, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Leave-one-origin-out raw scores; each origin scored by a model that
    never saw it (POLICY.md §5)."""
    scores = np.full(len(x), np.nan)
    for origin in pd.unique(groups):
        test = (groups == origin).to_numpy()
        train = ~test
        model = fit_model(kind, x.iloc[train], y.iloc[train])
        scores[test] = positive_probability(model, x.iloc[test], kind)
    if np.isnan(scores).any():
        raise AssertionError("OOF pass left unscored rows")
    return scores


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return iso


@dataclass(frozen=True)
class PairSums:
    """Per-pair absolute-error and |y| sums for one model's folds."""

    sums: pd.DataFrame  # index (series_id, forecast_origin): abs_err, abs_y

    def mean_category_wape(self) -> float:
        per_cat = self.sums.groupby(level="series_id")[["abs_err", "abs_y"]].sum()
        wapes = per_cat["abs_err"] / per_cat["abs_y"]
        if len(wapes) < 2:
            raise AssertionError("mean-category WAPE needs multiple categories")
        return float(wapes.mean())

    def category_wapes(self) -> pd.Series:
        per_cat = self.sums.groupby(level="series_id")[["abs_err", "abs_y"]].sum()
        return (per_cat["abs_err"] / per_cat["abs_y"]).rename("wape")


def pair_sums_from_folds(folds: pd.DataFrame, prediction_col: str = "prediction") -> PairSums:
    grouped = folds.groupby(["series_id", "forecast_origin"], sort=True)
    err = ((folds["actual"] - folds[prediction_col]).abs()).to_numpy()
    frame = pd.DataFrame(
        {
            "series_id": folds["series_id"].to_numpy(),
            "forecast_origin": pd.to_datetime(folds["forecast_origin"]).to_numpy(),
            "abs_err": err,
            "abs_y": folds["actual"].abs().to_numpy(),
        }
    )
    sums = frame.groupby(["series_id", "forecast_origin"], sort=True)[["abs_err", "abs_y"]].sum()
    return PairSums(sums=sums)


def blend_pair_sums(
    naive_folds: pd.DataFrame, theta_folds: pd.DataFrame, theta_weight: float
) -> PairSums:
    """Recorded global blend (0.5 theta + 0.5 naive per fold), not a new model:
    the reference weight comes from results/h3b_theta_blend.json."""
    keys = ["series_id", "forecast_origin", "horizon", "target_time"]
    merged = naive_folds.merge(
        theta_folds[keys + ["prediction"]], on=keys, suffixes=("_naive", "_theta")
    )
    if merged.empty:
        raise ValueError("cannot reconstruct blend: no shared folds")
    merged["prediction"] = (
        theta_weight * merged["prediction_theta"] + (1 - theta_weight) * merged["prediction_naive"]
    )
    return pair_sums_from_folds(merged)


def policy_pair_sums(selection: pd.Series, naive: PairSums, theta: PairSums) -> PairSums:
    """The switched policy's fold material: each pair contributes whichever
    model the policy chose for it."""
    n = naive.sums.rename(columns={"abs_err": "err_n", "abs_y": "y_n"})
    t = theta.sums.rename(columns={"abs_err": "err_t", "abs_y": "y_t"})
    joined = n.join(t, how="inner").join(selection.rename("choice"), how="inner")
    if len(joined) != len(selection):
        raise AssertionError("selection does not cover every pair")
    is_theta = (joined["choice"] == "theta").to_numpy()
    err = np.where(is_theta, joined["err_t"], joined["err_n"])
    y = np.where(is_theta, joined["y_t"], joined["y_n"])
    out = pd.DataFrame({"abs_err": err, "abs_y": y}, index=joined.index)
    return PairSums(sums=out)


@dataclass(frozen=True)
class SweepRow:
    tau: float
    policy_mean_cat_wape: float
    delta_vs_naive_pp: float
    theta_call_rate: float


def sweep_tau(
    probs: pd.Series,
    naive: PairSums,
    theta: PairSums,
    blend: PairSums,
    taus: np.ndarray = TAU_GRID,
    min_tau: float = 0.0,
    max_tau: float = 1.0,
) -> pd.DataFrame:
    """Full tau sweep of the switched policy on calibrated probabilities."""
    rows: list[SweepRow] = []
    naive_wape = naive.mean_category_wape()
    lo, hi = min(taus.min(), min_tau), max(taus.max(), max_tau)
    for tau in np.round(np.arange(lo, hi + 1e-9, 0.02), 4):
        selection = (probs >= tau).map({True: "theta", False: "naive"})
        policy = policy_pair_sums(selection, naive, theta)
        wape = policy.mean_category_wape()
        rows.append(
            SweepRow(
                tau=float(tau),
                policy_mean_cat_wape=wape,
                delta_vs_naive_pp=100.0 * (wape - naive_wape),
                theta_call_rate=float((selection == "theta").mean()),
            )
        )
    frame = pd.DataFrame(vars(r) for r in rows)
    frame.attrs["reference_mean_cat"] = {
        "naive": naive.mean_category_wape(),
        "theta": theta.mean_category_wape(),
        "blend": blend.mean_category_wape(),
    }
    return frame


def choose_tau(sweep: pd.DataFrame) -> float | None:
    """Minimize CHOOSE OOF policy mean-category WAPE among tau satisfying
    guardrail 1; ties resolve to the larger tau (closer to naive)."""
    eligible = sweep.loc[sweep["delta_vs_naive_pp"] <= GUARDRAIL1_MAX_LOSS_PP]
    if eligible.empty:
        return None
    best = eligible.loc[eligible["policy_mean_cat_wape"].idxmin()]
    tied = eligible.loc[
        np.isclose(eligible["policy_mean_cat_wape"], best["policy_mean_cat_wape"])
    ]
    return float(tied["tau"].max())


@dataclass(frozen=True)
class FrozenPolicy:
    """Everything needed to score a REPORT origin, frozen after CHOOSE."""

    kind: str
    model: object
    isotonic: IsotonicRegression
    tau: float
    series_categories: list[str]
    train_origins: list[pd.Timestamp]
    oof_brier: float
    choose_policy_wape: float
    sweep: pd.DataFrame = field(repr=False, default=None)

    def probabilities(self, table: pd.DataFrame) -> pd.Series:
        x = design_matrix(table, self.series_categories)
        raw = positive_probability(self.model, x, self.kind)
        return pd.Series(self.isotonic.predict(raw), index=table.index, name="p_theta")

    def selection(self, table: pd.DataFrame) -> pd.Series:
        probs = self.probabilities(table)
        return (probs >= self.tau).map({True: "theta", False: "naive"}).rename("choice")


def fit_policy(
    kind: str,
    train_table: pd.DataFrame,
    naive: PairSums,
    theta: PairSums,
    blend: PairSums,
    series_categories: list[str],
) -> FrozenPolicy:
    """Full CHOOSE-side procedure: OOF scores -> isotonic -> sweep -> tau ->
    final refit. Only CHOOSE data participates."""
    x = design_matrix(train_table, series_categories)
    y = train_table["theta_wins"].astype(bool)
    origins = pd.Series(train_table["forecast_origin"].to_numpy(), index=x.index)
    scores = oof_scores(kind, x, y, origins)
    iso = fit_isotonic(scores, y.to_numpy())
    oof_probs = pd.Series(iso.predict(scores), index=x.index)
    sweep = sweep_tau(oof_probs, naive, theta, blend)
    tau = choose_tau(sweep)
    if tau is None:
        raise ValueError("guardrail 1 unsatisfiable at every tau; POLICY_FINDINGS outcome")
    model = fit_model(kind, x, y)
    key = train_table.set_index(["series_id", "forecast_origin"]).index
    oof_brier = float(np.mean((oof_probs.to_numpy() - y.to_numpy()) ** 2))
    sel = (oof_probs >= tau).map({True: "theta", False: "naive"})
    sel.index = key
    choose_wape = policy_pair_sums(sel, naive, theta).mean_category_wape()
    return FrozenPolicy(
        kind=kind,
        model=model,
        isotonic=iso,
        tau=tau,
        series_categories=series_categories,
        train_origins=sorted(pd.unique(train_table["forecast_origin"]).tolist()),
        oof_brier=oof_brier,
        choose_policy_wape=choose_wape,
        sweep=sweep,
    )
