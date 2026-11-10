#!/usr/bin/env python3
"""Holdout scorecard: MASE, signed bias, category-by-horizon, origin-block
bootstrap CIs, and first-release re-scoring — a reporting-only reaggregation
of stored predictions.

Predeclared before recalculation (2026-11-10) and committed before the
numbers were produced:

1. Reporting only. No model is tuned, re-selected, or re-fitted; the
   promotion rule is unchanged; the gate verdict (M3_VINTAGE_BACKTEST.md)
   and the holdout verdict (FINDINGS.md section 3) stand exactly as
   recorded. This scorecard adds measurement around the frozen headline;
   nothing here feeds back into any decision.
2. Sources. Stored predictions only: ``reference/vintage_backtest_folds.csv``
   (seasonal naive + global LightGBM, the gate reference window 2017-10..
   2025-03; the LightGBM rows on holdout origins are labeled *previously
   observed* — those origins entered the record through the original gate
   run), ``results/folds_h5_theta.csv`` (theta, exploration origins), and
   ``results/folds_holdout_theta.csv`` (theta holdout rows recovered by
   ``scripts/holdout_theta_recovery.py`` under its verification gate). No
   prediction in any window is regenerated here.
3. Metric definitions (fixed before recalculation):
   a. Pooled WAPE = sum|y - yhat| / sum|y|; mean-category WAPE = equal-
      weighted mean of per-series WAPE (the gate's two conventions,
      unchanged).
   b. relMAE: per-series mean|e|_model / mean|e|_naive on identical folds,
      macro-averaged. Naive is exactly 1.00 by construction (the "naive
      MASE = 1.00" form of the request).
   c. MASE (Hyndman-Koehler, seasonal m=12): per fold |e| / s(series,
      origin), where s = mean |y_t - y_{t-12}| over the point-in-time
      training window (vintage as-of the origin: month-end + 45 days,
      observations at or before the fold's training cutoff). Macro MASE =
      equal-weighted mean of per-series means. Naive is approximately but
      not exactly 1.00 (training-scale denominator, not eval-scale).
   d. Signed bias: per-series MPE = mean((yhat - y)/y) x 100 and pooled
      relative bias = sum(yhat - y)/sum(y) x 100.
   e. Horizon: per-horizon mean-category WAPE plus the full
      per-(series, horizon) WAPE matrix, so long-range behavior cannot
      disappear inside an average.
   f. Uncertainty: origin-block bootstrap — whole forecast origins sampled
      with replacement (multiplicity preserved), B = 2000, seed 42,
      percentile 2.5/97.5 CIs for paired deltas vs naive on mean-category
      WAPE (pp), pooled WAPE (pp), and the categories-improved count.
      Windows reported with their origin count; the holdout has 8 blocks,
      so its CIs are coarse by construction and are reported as such.
   g. First-release re-scoring: the same stored predictions scored against
      each target month's first published vintage value (what could be
      evaluated operationally once the month was first released) alongside
      the stored latest-revision actuals (the eventual revised estimate).
      All WAPE/bias/MASE aggregations recomputed under each actual set;
      deltas vs naive recomputed under the same actuals.
4. Verification gates (STOP conditions, checked before anything is
   released):
   a. Naive on the holdout origins must reproduce the frozen single run's
      naive block (``results/holdout_theta.json``) to 1e-12.
   b. The recovered theta rows must reproduce the frozen candidate block
      (aggregates, by-series, by-horizon) to 1e-9, with exactly 352 rows on
      the frozen eight origins.
   c. The LightGBM holdout-slice mean-category WAPE must reproduce the
      recorded 5.64% gate slice (two decimals).
   d. Naive and LightGBM on the full gate window must reproduce the
      recorded gate numbers (8.16%/8.49% mean-category, 5.43%/5.48%
      pooled, two decimals) and the frozen summary's aggregate WAPEs to
      1e-9.
   e. Theta and naive on exploration origins must reproduce the frozen
      category win matrix's reference means to 1e-9.
   f. All three models' holdout means must agree with the predeclared
      three-model reaggregation (``results/holdout_three_model.json``).
   A mismatch stops the release; the more favorable number is never chosen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HOLDOUT_MIN_ORIGIN,
    fast_observations_as_of,
    load_panel,
    provenance,
    write_result,
)

ROOT = Path(__file__).resolve().parent.parent

GATE_FOLDS = ROOT / "reference" / "vintage_backtest_folds.csv"
THETA_EXPLORE = ROOT / "results" / "folds_h5_theta.csv"
THETA_HOLDOUT = ROOT / "results" / "folds_holdout_theta.csv"
FROZEN_RUN = ROOT / "results" / "holdout_theta.json"
THREE_MODEL = ROOT / "results" / "holdout_three_model.json"
WIN_MATRIX = ROOT / "results" / "category_win_matrix.json"
GATE_SUMMARY = ROOT / "reference" / "vintage_backtest_summary.json"
DATA_PATH = ROOT / "data" / "alfred_mrts_vintages.csv"

AGG_TOL = 1e-9
FROZEN_TOL = 1e-12
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
HORIZONS = ("1", "3", "6", "12")

PREDECLARED_STATEMENT = (
    "Reporting-only scorecard reaggregation, predeclared 2026-11-10 before "
    "recalculation: no tuning, no re-selection, no refit, promotion rule and "
    "both verdicts unchanged; stored predictions only (naive/LightGBM gate "
    "reference folds, theta exploration folds, verification-locked recovered "
    "theta holdout rows); metric definitions fixed as documented; six "
    "verification gates must pass or the release stops."
)

DATE_COLUMNS = [
    "forecast_origin",
    "target_time",
    "origin_as_of",
    "training_cutoff",
    "training_start",
    "data_vintage",
    "training_vintage_max",
    "actual_vintage_time",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stop(message: str) -> None:
    sys.exit(f"STOP: {message} Investigate before releasing this scorecard.")


def load_folds(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=DATE_COLUMNS)
    for col in ("series_id", "forecast_origin", "target_time", "horizon",
                "prediction", "actual", "model_name"):
        if col not in frame.columns:
            _stop(f"{path.name} is missing required column {col}.")
    return frame


def fold_keys(frame: pd.DataFrame) -> frozenset:
    return frozenset(
        zip(frame["series_id"], frame["forecast_origin"],
            frame["target_time"], frame["horizon"])
    )


def require_identical(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame) -> None:
    if fold_keys(a) != fold_keys(b):
        _stop(f"fold keys of {name_a} and {name_b} differ; paired metrics require "
              "identical (series, origin, target, horizon) folds.")


def first_release_actuals(panel: pd.DataFrame) -> pd.DataFrame:
    """First published value per (series, target month): the earliest vintage
    row containing that observation."""
    ordered = panel.sort_values(["vintage_time", "observation_time"])
    first = ordered.drop_duplicates(["series_id", "observation_time"], keep="first")
    return first[["series_id", "observation_time", "value"]].rename(
        columns={"value": "fr_actual"}
    )


def mase_scales(panel: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Point-in-time seasonal-naive scale s(series, origin): mean |y_t -
    y_{t-12}| over the vintage-as-of training window (Hyndman-Koehler
    denominator). The as-of rule (month-end + 45 days) is verified against
    every stored fold row's origin_as_of column before use."""
    rows = []
    for frame in frames:
        computed = (
            frame["forecast_origin"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)
        ).dt.tz_localize("UTC")
        stored = pd.to_datetime(frame["origin_as_of"], utc=True)
        if not (computed == stored).all():
            _stop("as-of rule (origin month-end + 45 days) does not match a "
                  "stored origin_as_of; refusing to derive MASE scales.")
        rows.append(
            frame[["series_id", "forecast_origin", "training_cutoff"]]
            .drop_duplicates()
        )
    keys = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["series_id", "forecast_origin"]
    )
    if keys.duplicated(["series_id", "forecast_origin"]).any():
        _stop("inconsistent training cutoffs across frames for one (series, origin).")

    scales = []
    for series_id, origin, cutoff in keys.itertuples(index=False):
        as_of = (origin + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45))
        obs = fast_observations_as_of(panel, pd.Timestamp(as_of, tz="UTC"))
        window = obs[
            (obs["series_id"] == series_id)
            & (obs["observation_time"] <= cutoff)
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
        scales.append(
            {"series_id": series_id, "forecast_origin": origin, "mase_scale": scale}
        )
    return pd.DataFrame(scales)


def _attach(frame: pd.DataFrame, scales: pd.DataFrame,
            first_release: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(scales, on=["series_id", "forecast_origin"], how="left", validate="many_to_one")
    out = out.merge(
        first_release,
        left_on=["series_id", "target_time"],
        right_on=["series_id", "observation_time"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["observation_time"])
    if out["mase_scale"].isna().any() or out["fr_actual"].isna().any():
        _stop("MASE scale or first-release actual coverage is incomplete after merge.")
    return out


def _wape(actual: pd.Series, prediction: pd.Series) -> float:
    denom = float(actual.abs().sum())
    if denom == 0:
        _stop("WAPE undefined: actuals sum to zero in a block.")
    return float((prediction - actual).abs().sum() / denom)


def model_stats(frame: pd.DataFrame, actual_col: str) -> dict:
    act = frame[actual_col]
    pred = frame["prediction"]
    err = pred - act
    abs_err = err.abs()

    per_series = {}
    for series_id, group in frame.groupby("series_id", sort=True):
        per_series[str(series_id)] = {
            "wape": _wape(group[actual_col], group["prediction"]),
            "mae": float((group["prediction"] - group[actual_col]).abs().mean()),
            "mase": float(
                ((group["prediction"] - group[actual_col]).abs()
                 / group["mase_scale"]).mean()
            ),
            "mpe_pp": float(
                (((group["prediction"] - group[actual_col]) / group[actual_col]).mean())
                * 100
            ),
        }

    by_horizon_mean_category = {}
    by_horizon_pooled = {}
    by_series_horizon: dict[str, dict[str, float]] = {}
    for horizon, group in frame.groupby("horizon", sort=True):
        series_wapes = {
            str(s): _wape(g[actual_col], g["prediction"])
            for s, g in group.groupby("series_id", sort=True)
        }
        by_horizon_mean_category[str(int(horizon))] = float(
            pd.Series(series_wapes).mean()
        )
        # pooled WAPE per horizon — the frozen run's by_horizon convention
        by_horizon_pooled[str(int(horizon))] = _wape(group[actual_col], group["prediction"])
        for series, value in series_wapes.items():
            by_series_horizon.setdefault(series, {})[str(int(horizon))] = value

    return {
        "fold_rows": int(len(frame)),
        "actual_set": actual_col,
        "pooled_wape": _wape(act, pred),
        "mean_category_wape": float(pd.Series({s: b["wape"] for s, b in per_series.items()}).mean()),
        "mase_macro": float(pd.Series({s: b["mase"] for s, b in per_series.items()}).mean()),
        "pooled_relative_bias_pp": float(err.sum() / act.sum() * 100),
        "by_series": per_series,
        "by_horizon_mean_category_wape": by_horizon_mean_category,
        "by_horizon_pooled_wape": by_horizon_pooled,
        "by_series_horizon_wape": by_series_horizon,
    }


def rel_mae_vs_naive(model: dict, naive: dict) -> tuple[dict, float]:
    ratios = {
        series: model["by_series"][series]["mae"] / naive["by_series"][series]["mae"]
        for series in sorted(model["by_series"])
    }
    macro = float(pd.Series(ratios).mean())
    return ratios, macro


def _pivot_sums(frame: pd.DataFrame, actual_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.assign(_abs_err=(frame["prediction"] - frame[actual_col]).abs())
    se = work.pivot_table(index="forecast_origin", columns="series_id",
                          values="_abs_err", aggfunc="sum")
    sy = work.pivot_table(index="forecast_origin", columns="series_id",
                          values=actual_col, aggfunc="sum")
    if not se.index.equals(sy.index) or list(se.columns) != list(sy.columns):
        _stop("origin/series pivots misaligned within a model.")
    return se, sy


def bootstrap_delta(candidate: pd.DataFrame, naive: pd.DataFrame, actual_col: str) -> dict:
    """Origin-block bootstrap (whole origins, with multiplicity, B=2000,
    seed 42, percentile 2.5/97.5) for paired deltas vs naive."""
    se_c, sy_c = _pivot_sums(candidate, actual_col)
    se_n, sy_n = _pivot_sums(naive, actual_col)
    if not (se_c.index.equals(se_n.index)):
        _stop("bootstrap origin grids differ between candidate and naive.")
    se_c_v, sy_c_v = se_c.to_numpy(), sy_c.to_numpy()
    se_n_v, sy_n_v = se_n.to_numpy(), sy_n.to_numpy()
    n = len(se_c_v)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    point_wc = se_c_v.sum(axis=0) / sy_c_v.sum(axis=0)
    point_wn = se_n_v.sum(axis=0) / sy_n_v.sum(axis=0)
    point = {
        "mean_category_delta_pp": float(100 * (point_wn.mean() - point_wc.mean())),
        "pooled_delta_pp": float(
            100 * (se_n_v.sum() / sy_n_v.sum() - se_c_v.sum() / sy_c_v.sum())
        ),
        "categories_improved": int((point_wc < point_wn).sum()),
    }
    deltas_mc, deltas_pooled, counts = [], [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        wc = se_c_v[idx].sum(axis=0) / sy_c_v[idx].sum(axis=0)
        wn = se_n_v[idx].sum(axis=0) / sy_n_v[idx].sum(axis=0)
        deltas_mc.append(100 * (wn.mean() - wc.mean()))
        deltas_pooled.append(
            100 * (se_n_v[idx].sum() / sy_n_v[idx].sum()
                   - se_c_v[idx].sum() / sy_c_v[idx].sum())
        )
        counts.append(int((wc < wn).sum()))

    def ci(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        lo, hi = np.percentile(arr, CI_PERCENTILES)
        return {"lo": float(lo), "hi": float(hi)}

    return {
        "n_origins": int(n),
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "forecast_origin (whole-origin blocks, with multiplicity)",
        "point": point,
        "mean_category_delta_pp_ci": ci(deltas_mc),
        "pooled_delta_pp_ci": ci(deltas_pooled),
        "categories_improved_ci": ci([float(c) for c in counts]),
    }


def main() -> None:
    frozen = json.loads(FROZEN_RUN.read_text())
    three_model = json.loads(THREE_MODEL.read_text())
    win_matrix = json.loads(WIN_MATRIX.read_text())
    gate_summary = json.loads(GATE_SUMMARY.read_text())

    gate = load_folds(GATE_FOLDS)
    theta_exp = load_folds(THETA_EXPLORE)
    theta_hold = load_folds(THETA_HOLDOUT)

    naive_all = gate[gate["model_name"] == "seasonal_naive"].copy()
    lgbm_all = gate[gate["model_name"] == "global_lightgbm"].copy()
    exploration = naive_all["forecast_origin"] < HOLDOUT_MIN_ORIGIN
    naive_exp = naive_all[exploration].copy()
    naive_hold = naive_all[~exploration].copy()
    lgbm_exp = lgbm_all[exploration].copy()
    lgbm_hold = lgbm_all[~exploration].copy()

    frozen_origins = [pd.Timestamp(o) for o in frozen["holdout_origins"]]
    if sorted(theta_hold["forecast_origin"].unique()) != sorted(frozen_origins):
        _stop("recovered theta holdout origins do not equal the frozen eight.")
    if len(theta_hold) != 352:
        _stop("recovered theta holdout rows are not exactly 352.")

    require_identical("exploration naive", naive_exp, "exploration theta", theta_exp)
    require_identical("holdout naive", naive_hold, "holdout theta", theta_hold)
    require_identical("holdout naive", naive_hold, "holdout LightGBM", lgbm_hold)
    require_identical("gate naive", naive_all, "gate LightGBM", lgbm_all)

    panel = load_panel()
    scales = mase_scales(panel, [naive_all, lgbm_all, theta_exp, theta_hold])
    first_release = first_release_actuals(panel)

    prepared = {
        "naive_all": _attach(naive_all, scales, first_release),
        "lgbm_all": _attach(lgbm_all, scales, first_release),
        "theta_exp": _attach(theta_exp, scales, first_release),
        "theta_hold": _attach(theta_hold, scales, first_release),
        "naive_exp": _attach(naive_exp, scales, first_release),
        "naive_hold": _attach(naive_hold, scales, first_release),
        "lgbm_hold": _attach(lgbm_hold, scales, first_release),
    }

    # ---- verification gates (all must pass before any release) ----
    verification: dict[str, object] = {}

    def check(label: str, recomputed: float, frozen_value: float, tol: float) -> None:
        diff = abs(recomputed - frozen_value)
        if diff > tol:
            _stop(f"gate '{label}' failed: recomputed {recomputed!r} vs frozen "
                  f"{frozen_value!r} (diff {diff:.3e} > {tol:.0e}).")
        verification[label] = diff

    hold_stats_naive = model_stats(prepared["naive_hold"], "actual")
    hold_stats_theta = model_stats(prepared["theta_hold"], "actual")
    hold_stats_lgbm = model_stats(prepared["lgbm_hold"], "actual")
    gate_stats_naive = model_stats(prepared["naive_all"], "actual")
    gate_stats_lgbm = model_stats(prepared["lgbm_all"], "actual")
    exp_stats_theta = model_stats(prepared["theta_exp"], "actual")
    exp_stats_naive = model_stats(prepared["naive_exp"], "actual")

    check("a.naive_hold.mean_category", hold_stats_naive["mean_category_wape"],
          frozen["naive"]["mean_category_wape"], FROZEN_TOL)
    check("a.naive_hold.aggregate", hold_stats_naive["pooled_wape"],
          frozen["naive"]["aggregate_wape"], FROZEN_TOL)
    for series, block in frozen["naive"]["by_series"].items():
        check(f"a.naive_hold.{series}", hold_stats_naive["by_series"][series]["wape"],
              block["wape"], FROZEN_TOL)
    check("b.theta_hold.mean_category", hold_stats_theta["mean_category_wape"],
          frozen["candidate"]["mean_category_wape"], AGG_TOL)
    check("b.theta_hold.aggregate", hold_stats_theta["pooled_wape"],
          frozen["candidate"]["aggregate_wape"], AGG_TOL)
    for series, block in frozen["candidate"]["by_series"].items():
        check(f"b.theta_hold.{series}", hold_stats_theta["by_series"][series]["wape"],
              block["wape"], AGG_TOL)
    for horizon, value in frozen["by_horizon"]["theta"].items():
        check(f"b.theta_hold.h{horizon}",
              hold_stats_theta["by_horizon_pooled_wape"][horizon], value, AGG_TOL)
    for horizon, value in frozen["by_horizon"]["naive"].items():
        check(f"a.naive_hold.h{horizon}",
              hold_stats_naive["by_horizon_pooled_wape"][horizon], value, FROZEN_TOL)
    if round(hold_stats_lgbm["mean_category_wape"] * 100, 2) != 5.64:
        _stop("gate 'c' failed: LightGBM holdout-slice mean-category WAPE does not "
              "reproduce the recorded 5.64% slice.")
    verification["c.lgbm_hold.slice_5_64"] = round(
        hold_stats_lgbm["mean_category_wape"] * 100, 2
    )
    check("d.naive_gate.mean_category", gate_stats_naive["mean_category_wape"], 0.0816, 5e-5)
    check("d.lgbm_gate.mean_category", gate_stats_lgbm["mean_category_wape"], 0.0849, 5e-5)
    check("d.naive_gate.pooled_2dp", round(gate_stats_naive["pooled_wape"] * 100, 2),
          5.43, 0.0)
    check("d.lgbm_gate.pooled_2dp", round(gate_stats_lgbm["pooled_wape"] * 100, 2),
          5.48, 0.0)
    check("d.naive_gate.aggregate_vs_summary", gate_stats_naive["pooled_wape"],
          gate_summary["seasonal_naive"]["aggregate"]["wape"], AGG_TOL)
    check("d.lgbm_gate.aggregate_vs_summary", gate_stats_lgbm["pooled_wape"],
          gate_summary["global_lightgbm"]["aggregate"]["wape"], AGG_TOL)
    fe_refs = win_matrix["windows"]["full_exploration"]["mean_category_wape_for_reference"]
    check("e.theta_explore.reference", exp_stats_theta["mean_category_wape"],
          fe_refs["theta"], AGG_TOL)
    check("e.naive_explore.reference", exp_stats_naive["mean_category_wape"],
          fe_refs["naive"], AGG_TOL)
    for row in three_model["rows"]:
        model_map = {
            "seasonal_naive": hold_stats_naive,
            "theta_method": hold_stats_theta,
            "global_lightgbm": hold_stats_lgbm,
        }
        stats = model_map[row["model_name"]]
        check(f"f.three_model.{row['model_name']}.mean_category",
              stats["mean_category_wape"], row["mean_category_wape"], FROZEN_TOL)

    # ---- scorecard assembly ----
    naive_series_mae = {
        "holdout": {s: b["mae"] for s, b in hold_stats_naive["by_series"].items()},
        "gate_full": {s: b["mae"] for s, b in gate_stats_naive["by_series"].items()},
        "exploration": {s: b["mae"] for s, b in exp_stats_naive["by_series"].items()},
    }

    def window_block(label: str, models: dict[str, dict], n_origins: int,
                     caveat: str | None = None) -> dict:
        out_models = {}
        for name, stats in models.items():
            block = dict(stats)
            if name != "seasonal_naive":
                ratios, macro = rel_mae_vs_naive(stats, {
                    "by_series": {s: {"mae": m} for s, m in
                                  naive_series_mae[label].items()}
                })
                block["rel_mae_by_series"] = ratios
                block["rel_mae_macro"] = macro
            else:
                block["rel_mae_macro"] = 1.0
            block["categories_beating_naive"] = (
                sum(1 for s in block["by_series"]
                    if block["by_series"][s]["wape"]
                    < naive_map[label]["by_series"][s]["wape"])
            )
            out_models[name] = block
        return {
            "n_origins": n_origins,
            "models": out_models,
            **({"caveat": caveat} if caveat else {}),
        }

    naive_map = {
        "holdout": hold_stats_naive,
        "gate_full": gate_stats_naive,
        "exploration": exp_stats_naive,
    }

    windows = {
        "holdout": window_block(
            "holdout",
            {
                "seasonal_naive": hold_stats_naive,
                "theta_method": hold_stats_theta,
                "global_lightgbm": hold_stats_lgbm,
            },
            n_origins=8,
            caveat=(
                "8 origin blocks — bootstrap CIs here are coarse. LightGBM rows "
                "are previously observed (gate reference folds), not an "
                "untouched holdout result. Theta rows are the recovered, "
                "verification-locked single-run predictions."
            ),
        ),
        "gate_full": window_block(
            "gate_full",
            {"seasonal_naive": gate_stats_naive, "global_lightgbm": gate_stats_lgbm},
            n_origins=int(naive_all["forecast_origin"].nunique()),
            caveat=(
                "Full gate reference window (2017-10..2025-03, all previously "
                "observed folds). Includes the holdout origins in its average."
            ),
        ),
        "exploration": window_block(
            "exploration",
            {"seasonal_naive": exp_stats_naive, "theta_method": exp_stats_theta},
            n_origins=int(theta_exp["forecast_origin"].nunique()),
        ),
    }

    boots = {
        "holdout": {
            "theta_vs_naive": bootstrap_delta(prepared["theta_hold"], prepared["naive_hold"], "actual"),
            "lightgbm_vs_naive": bootstrap_delta(prepared["lgbm_hold"], prepared["naive_hold"], "actual"),
            "theta_vs_lightgbm": bootstrap_delta(prepared["theta_hold"], prepared["lgbm_hold"], "actual"),
        },
        "gate_full": {
            "lightgbm_vs_naive": bootstrap_delta(prepared["lgbm_all"], prepared["naive_all"], "actual"),
        },
        "exploration": {
            "theta_vs_naive": bootstrap_delta(prepared["theta_exp"], prepared["naive_exp"], "actual"),
        },
    }

    # First-release re-scoring: same stored predictions, earliest-vintage actuals.
    fr_stats = {
        "holdout": {
            "seasonal_naive": model_stats(prepared["naive_hold"], "fr_actual"),
            "theta_method": model_stats(prepared["theta_hold"], "fr_actual"),
            "global_lightgbm": model_stats(prepared["lgbm_hold"], "fr_actual"),
        },
        "gate_full": {
            "seasonal_naive": model_stats(prepared["naive_all"], "fr_actual"),
            "global_lightgbm": model_stats(prepared["lgbm_all"], "fr_actual"),
        },
    }
    fr_bootstrap = {
        "holdout": {
            "theta_vs_naive": bootstrap_delta(prepared["theta_hold"], prepared["naive_hold"], "fr_actual"),
            "lightgbm_vs_naive": bootstrap_delta(prepared["lgbm_hold"], prepared["naive_hold"], "fr_actual"),
        },
        "gate_full": {
            "lightgbm_vs_naive": bootstrap_delta(prepared["lgbm_all"], prepared["naive_all"], "fr_actual"),
        },
    }

    payload = {
        "provenance": provenance({"purpose": "reporting-only holdout scorecard"}),
        "predeclared_statement": PREDECLARED_STATEMENT,
        "metric_definitions": {
            "pooled_wape": "sum|y-yhat| / sum|y|",
            "mean_category_wape": "equal-weighted mean of per-series WAPE",
            "rel_mae": "per-series MAE(model)/MAE(naive) on identical folds, macro-averaged; naive = 1.00",
            "mase": "Hyndman-Koehler seasonal m=12; scale = mean|y_t - y_{t-12}| over the point-in-time training window (vintage as-of origin month-end + 45 days, observations <= training cutoff)",
            "signed_bias": "per-series MPE (mean percentage error, pp) and pooled relative bias (pp)",
            "uncertainty": "origin-block bootstrap, whole origins with multiplicity, B=2000, seed=42, percentile 2.5/97.5",
            "first_release": "actual = target month's first published vintage value; stored actual = latest revision in archive",
        },
        "sources": {
            "gate_folds": {"path": "reference/vintage_backtest_folds.csv", "sha256": _sha256(GATE_FOLDS)},
            "theta_exploration": {"path": "results/folds_h5_theta.csv", "sha256": _sha256(THETA_EXPLORE)},
            "theta_holdout_recovered": {"path": "results/folds_holdout_theta.csv", "sha256": _sha256(THETA_HOLDOUT)},
            "frozen_run": "results/holdout_theta.json",
            "three_model_reaggregation": "results/holdout_three_model.json",
        },
        "verification_gates": verification,
        "windows": windows,
        "bootstrap": boots,
        "first_release": {
            "note": "Same stored predictions; actuals replaced by first published vintage values. Final-vintage counterparts are the 'actual' blocks above.",
            "windows": fr_stats,
            "bootstrap": fr_bootstrap,
        },
    }
    write_result("holdout_scorecard", payload)

    md_lines = render_markdown(windows, boots, fr_stats, fr_bootstrap)
    md_path = ROOT / "results" / "holdout_scorecard.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {md_path.relative_to(ROOT)}")

    h = windows["holdout"]["models"]
    print("\nverification: all gates passed; headline block (holdout, final vintages):")
    for name in ("seasonal_naive", "theta_method", "global_lightgbm"):
        b = h[name]
        print(f"  {name:<16} mean-cat {b['mean_category_wape']*100:5.2f}%  "
              f"MASE {b['mase_macro']:5.3f}  relMAE {b['rel_mae_macro']:5.3f}  "
              f"bias {b['pooled_relative_bias_pp']:+6.2f}pp  "
              f"beats naive {b['categories_beating_naive']}/11")
    ci = boots["holdout"]["theta_vs_naive"]
    print(f"  theta vs naive mean-category delta {ci['point']['mean_category_delta_pp']:+.2f}pp "
          f"(origin-block CI {ci['mean_category_delta_pp_ci']['lo']:+.2f}..{ci['mean_category_delta_pp_ci']['hi']:+.2f}, "
          f"8 blocks)")


def render_markdown(windows: dict, boots: dict, fr_stats: dict, fr_bootstrap: dict) -> list[str]:
    lines = [
        "# Holdout scorecard (reporting-only reaggregation, 2026-11-10)",
        "",
        "Predeclared metric definitions and verification gates in",
        "`scripts/holdout_scorecard.py`; every gate reproduces the frozen record",
        "before release. All numbers two decimals unless noted.",
        "",
    ]

    def model_table(window: str) -> None:
        block = windows[window]
        lines.append(f"### {window} ({block['n_origins']} origins)")
        lines.append("")
        if "caveat" in block:
            lines.append(f"*{block['caveat']}*")
            lines.append("")
        lines.append("| model | pooled WAPE | mean-cat WAPE | relMAE | MASE | bias (pp) | beats naive | h=1 | h=3 | h=6 | h=12 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, stats in block["models"].items():
            hz = stats["by_horizon_mean_category_wape"]
            lines.append(
                f"| {name} | {stats['pooled_wape']*100:.2f}% | "
                f"{stats['mean_category_wape']*100:.2f}% | {stats['rel_mae_macro']:.3f} | "
                f"{stats['mase_macro']:.3f} | {stats['pooled_relative_bias_pp']:+.2f} | "
                f"{stats['categories_beating_naive']}/11 | "
                + " | ".join(f"{hz[h]*100:.2f}%" for h in HORIZONS) + " |"
            )
        lines.append("")

    for window in ("holdout", "gate_full", "exploration"):
        model_table(window)

    lines.append("### Origin-block bootstrap CIs (paired deltas vs naive unless labeled)")
    lines.append("")
    lines.append("| window | pair | mean-category delta (pp) | CI | pooled delta (pp) | CI | categories improved |")
    lines.append("|---|---|---:|---|---:|---|---:|")
    for window, pairs in boots.items():
        for pair, boot in pairs.items():
            p = boot["point"]
            mc = boot["mean_category_delta_pp_ci"]
            pl = boot["pooled_delta_pp_ci"]
            cnt = boot["categories_improved_ci"]
            lines.append(
                f"| {window} | {pair} | {p['mean_category_delta_pp']:+.2f} | "
                f"{mc['lo']:+.2f}..{mc['hi']:+.2f} | {p['pooled_delta_pp']:+.2f} | "
                f"{pl['lo']:+.2f}..{pl['hi']:+.2f} | "
                f"{p['categories_improved']:.0f} ({cnt['lo']:.1f}..{cnt['hi']:.1f}) |"
            )
    lines.append("")
    lines.append("*The holdout window has 8 origin blocks; its CIs are coarse by "
                 "construction and should be read as scale-of-evidence, not as "
                 "significance tests.*")
    lines.append("")

    lines.append("### First-release vs final-vintage actuals (same stored predictions)")
    lines.append("")
    lines.append("| window | model | mean-cat WAPE (first release) | mean-cat WAPE (final) | pooled WAPE (first release) | pooled WAPE (final) | bias first release (pp) | bias final (pp) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for window, models in fr_stats.items():
        final_models = windows[window]["models"]
        for name, stats in models.items():
            fin = final_models[name]
            lines.append(
                f"| {window} | {name} | {stats['mean_category_wape']*100:.2f}% | "
                f"{fin['mean_category_wape']*100:.2f}% | {stats['pooled_wape']*100:.2f}% | "
                f"{fin['pooled_wape']*100:.2f}% | {stats['pooled_relative_bias_pp']:+.2f} | "
                f"{fin['pooled_relative_bias_pp']:+.2f} |"
            )
    lines.append("")
    lines.append("### First-release bootstrap (holdout)")
    lines.append("")
    for pair, boot in fr_bootstrap["holdout"].items():
        p = boot["point"]
        mc = boot["mean_category_delta_pp_ci"]
        lines.append(
            f"- {pair}: mean-category delta {p['mean_category_delta_pp']:+.2f}pp "
            f"(CI {mc['lo']:+.2f}..{mc['hi']:+.2f})"
        )
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
