"""H7 direct multi-horizon models (predeclared in H7_FEATURE_LADDER.md §4).

For horizon h, training rows are (feature month u, target month u+h-1) with
features computed from values strictly before u and target
``log(v_{u+h-1} / v_{u+h-13})``. At an origin O the prediction row is
u = O+1, so the freshest known value v_O enters as lag_1 — and for h = 1 the
rows, features, target, parameters, and reconstruction are *exactly* the
gate point model, which is the predeclared sanity anchor (§7.1: direct-base
h=1 must reproduce the gate reference folds' h=1 predictions).

Reconstruction multiplies the predicted log-ratio by the origin-known
``v_{O+h-12}`` (for h <= 12 this month is <= O, hence published at the
origin as-of). No recursion anywhere: each horizon is its own model.

Vintage safety: everything derives from the harness's as-of snapshot plus
the archived vintage panel (revision context, H2-exact) and the archived
external vintage panel (origin-constant context, H2 precedent). Nothing is
read from any later vintage or today's download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet

from vintageforensics.feature_groups import (
    GROUP_FEATURES,
    add_base_ratios,
    attach_revision_columns,
    base_features,
    cross_features,
    external_context_as_of,
    growth_features,
    revision_context_as_of,
    trend_features,
)
from vintageforensics.features import validate_monthly_panel

GATE_PARAMS = {
    # PROTOCOL invariant 4: the gate LightGBM parameters stay fixed everywhere.
    "n_estimators": 250,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 10,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

ELASTIC_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1)
ELASTIC_L1_RATIOS = (0.2, 0.8)
ELASTIC_INNER_VAL_MONTHS = 18
ELASTIC_INNER_FOLDS = 4
ELASTIC_MIN_TRAIN_MONTHS = 60
ELASTIC_DEFAULT = (0.01, 0.2)


def _month_number(month: pd.Timestamp) -> int:
    return month.year * 12 + month.month


class _DirectBase:
    """Shared row construction for the direct models (horizons 1/3/6/12)."""

    horizons = (1, 3, 6, 12)

    def __init__(self, *, version: str, groups: tuple[str, ...]) -> None:
        unknown = set(groups) - set(GROUP_FEATURES)
        if unknown:
            raise ValueError(f"unknown feature groups: {sorted(unknown)}")
        if "base" not in groups:
            raise ValueError("the base group is always enabled")
        self.version = version
        self.groups = tuple(groups)
        self._origin_as_of: pd.Timestamp | None = None
        self._design: pd.DataFrame | None = None
        self._values: dict[tuple[str, pd.Timestamp], float] = {}
        self._origin_month: pd.Timestamp | None = None
        self._series_ids: list[str] = []
        self._prediction_month: pd.Timestamp | None = None

    def set_origin_context(self, origin_as_of) -> None:
        self._origin_as_of = pd.Timestamp(origin_as_of)

    def _require_context(self) -> None:
        if self._origin_as_of is None:
            raise RuntimeError("origin context missing; harness origin_hook must run before fit")

    def _prepare(self, history: pd.DataFrame) -> None:
        self._require_context()
        required = {"series_id", "observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        panel = validate_monthly_panel(history.loc[:, ["series_id", "observation_time", "value"]])
        values = pd.to_numeric(panel["value"], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("training values must be finite positive numbers")
        panel["value"] = values.astype(float)
        self._series_ids = sorted(panel["series_id"].astype(str).unique())
        series_codes = {series_id: index for index, series_id in enumerate(self._series_ids)}
        self._origin_month = pd.Timestamp(panel["observation_time"].max())
        self._prediction_month = self._origin_month + pd.offsets.MonthBegin(1)
        self._values = {
            (str(series_id), pd.Timestamp(month)): float(value)
            for series_id, month, value in panel.loc[:, ["series_id", "observation_time", "value"]].itertuples(index=False)
        }
        working = pd.concat(
            [
                panel.loc[:, ["series_id", "observation_time", "value"]],
                pd.DataFrame(
                    {
                        "series_id": self._series_ids,
                        "observation_time": [self._prediction_month] * len(self._series_ids),
                        "value": np.nan,
                    }
                ),
            ],
            ignore_index=True,
        )
        featured = base_features(working, series_codes)
        for group in ("growth", "trend", "cross"):
            if group in self.groups:
                builder = {"growth": growth_features, "trend": trend_features, "cross": cross_features}[group]
                extra = builder(working)
                featured = featured.merge(
                    extra.loc[:, ["series_id", "observation_time", *GROUP_FEATURES[group]]],
                    on=["series_id", "observation_time"],
                    how="left",
                    validate="one_to_one",
                )
        if "revision" in self.groups:
            if getattr(self, "_vintage_panel", None) is None:
                raise ValueError("revision features require the archived vintage panel")
            snapshot = panel.assign(series_id=panel["series_id"].astype(str))
            rev_context, vintage_age_days, moy_lookup = revision_context_as_of(
                self._vintage_panel, snapshot, self._origin_as_of
            )
            featured = attach_revision_columns(
                featured, rev_context, vintage_age_days, self._origin_month, moy_lookup
            )
        if "external" in self.groups:
            if getattr(self, "_external_panel", None) is None:
                raise ValueError("external features require the external vintage panel")
            context = external_context_as_of(self._external_panel, self._origin_as_of, self._origin_month)
            for column, value in context.items():
                featured[column] = value
        expected = [column for group in self.groups for column in GROUP_FEATURES[group]]
        missing = sorted(set(expected) - set(featured.columns))
        if missing:
            raise ValueError(f"feature construction left columns missing: {missing}")
        self._design = featured

    def _horizon_rows(self, horizon: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Training (X, y) and the prediction row for one horizon."""
        design = self._design
        offsets = horizon - 1
        month_numbers = (
            design["observation_time"].dt.year * 12 + design["observation_time"].dt.month
        )
        target_number = month_numbers + offsets
        origin_number = _month_number(self._origin_month)
        start_number = month_numbers.iloc[0]
        y = np.full(len(design), np.nan)
        values_array = design["value"].to_numpy(dtype=float)
        for position, number in enumerate(target_number.to_numpy()):
            if number > origin_number or number - 12 < start_number:
                continue
            row = design.iloc[position]
            key_t = (str(row["series_id"]), self._number_to_month(number))
            key_lag = (str(row["series_id"]), self._number_to_month(number - 12))
            if key_t in self._values and key_lag in self._values:
                y[position] = np.log(self._values[key_t] / self._values[key_lag])
        feature_columns = [column for group in self.groups for column in GROUP_FEATURES[group]]
        complete = design[feature_columns].notna().all(axis=1).to_numpy() & np.isfinite(y)
        train = design.loc[complete]
        prediction_row = design.loc[design["observation_time"] == self._prediction_month]
        if len(prediction_row) != len(self._series_ids):
            raise RuntimeError("prediction row construction failed")
        return train, y[complete], prediction_row

    def _number_to_month(self, number: int) -> pd.Timestamp:
        # inverse of year*12 + month (month 1..12): divmod(number-1, 12)
        year, month_index = divmod(number - 1, 12)
        return pd.Timestamp(year=year, month=month_index + 1, day=1)

    def _reconstruction_value(self, series_id: str, horizon: int) -> float:
        month = self._number_to_month(_month_number(self._origin_month) + horizon - 12)
        return self._values[(series_id, month)]

    def _predictions(self, per_horizon_predictions) -> pd.DataFrame:
        """Assemble fold rows from per-horizon log-ratio arrays (aligned with
        ``self._series_ids``, the design's sorted prediction-row order)."""
        records: list[dict[str, object]] = []
        for horizon in self.horizons:
            log_ratios = per_horizon_predictions(horizon)
            for index, series_id in enumerate(self._series_ids):
                records.append(
                    {
                        "series_id": series_id,
                        "target_time": self._number_to_month(
                            _month_number(self._origin_month) + horizon
                        ),
                        "horizon": horizon,
                        "prediction": float(
                            self._reconstruction_value(series_id, horizon)
                            * np.exp(log_ratios[index])
                        ),
                    }
                )
        return pd.DataFrame.from_records(records)

    def _prediction_rows(self) -> pd.DataFrame:
        rows = self._design.loc[
            self._design["observation_time"] == self._prediction_month
        ].reset_index(drop=True)
        if rows["series_id"].astype(str).tolist() != self._series_ids:
            raise RuntimeError("prediction row order does not match the series registry")
        return rows


class DirectLightGBM(_DirectBase):
    """Direct per-horizon LightGBM on the predeclared feature groups.

    Gate hyperparameters byte-for-byte (PROTOCOL invariant 4); the h=1 model
    is the gate point model (predeclared sanity anchor).
    """

    name = "direct_lightgbm"

    def __init__(self, *, version: str, groups: tuple[str, ...] = ("base",), vintage_panel: pd.DataFrame | None = None, external_panel: pd.DataFrame | None = None) -> None:
        super().__init__(version=version, groups=groups)
        self._vintage_panel = vintage_panel
        self._external_panel = external_panel
        self._models: dict[int, LGBMRegressor] = {}

    def fit(self, history: pd.DataFrame) -> "DirectLightGBM":
        self._prepare(history)
        feature_columns = [column for group in self.groups for column in GROUP_FEATURES[group]]
        for horizon in self.horizons:
            train, y, _ = self._horizon_rows(horizon)
            if train.empty:
                raise ValueError(f"no training rows remain for horizon {horizon}")
            x_train = train.loc[:, feature_columns]
            model = LGBMRegressor(**GATE_PARAMS, objective="regression_l1")
            self._models[horizon] = model.fit(
                x_train, y, categorical_feature=["series_code"]
            )
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("fit must be called before predict")
        if horizon != max(self.horizons):
            raise ValueError("direct models are fit for the full horizon set")
        feature_columns = [column for group in self.groups for column in GROUP_FEATURES[group]]
        prediction_rows = self._prediction_rows()

        def per_horizon_predictions(h: int) -> np.ndarray:
            return self._models[h].predict(prediction_rows.loc[:, feature_columns])

        return self._predictions(per_horizon_predictions)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": dict(GATE_PARAMS),
            "groups": list(self.groups),
            "features": [c for g in self.groups for c in GROUP_FEATURES[g]],
            "series_ids": list(self._series_ids),
        }


class DirectElasticNet(_DirectBase):
    """Direct per-horizon elastic net with per-origin inner-CV regularization
    (H7_FEATURE_LADDER.md §4 item 3, §6: selection inside the backtest).

    Design: standardized continuous features + series one-hot (drop the
    alphabetically first series as reference) + month-of-feature-month
    one-hot (drop January). Inner CV: expanding window, four validation
    blocks of 18 target months counted back from the training edge; a block
    whose training side has < 60 distinct months is skipped; selection
    metric is pooled MAE of log-ratio residuals; ties resolve to the
    smaller alpha, then the smaller l1_ratio; if every block is skipped the
    predeclared default (0.01, 0.2) applies. No randomness anywhere.
    """

    name = "direct_elastic_net"

    def __init__(self, *, version: str, groups: tuple[str, ...] = ("base",), vintage_panel: pd.DataFrame | None = None, external_panel: pd.DataFrame | None = None) -> None:
        super().__init__(version=version, groups=groups)
        self._vintage_panel = vintage_panel
        self._external_panel = external_panel
        self._models: dict[int, ElasticNet] = {}
        self._apply_matrices: dict[int, np.ndarray] = {}
        self._n_continuous = 0

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        continuous = [
            column for group in self.groups for column in GROUP_FEATURES[group]
            if column != "series_code"
        ]
        x = rows.loc[:, continuous].to_numpy(dtype=float)
        series_dummies = np.column_stack([
            (rows["series_id"].astype(str) == series_id).to_numpy(dtype=float)
            for series_id in self._series_ids[1:]  # drop reference series
        ])
        months = rows["observation_time"].dt.month.to_numpy()
        month_dummies = np.column_stack([
            (months == month).astype(float) for month in range(2, 13)  # drop January
        ])
        return np.hstack([x, series_dummies, month_dummies])

    def fit(self, history: pd.DataFrame) -> "DirectElasticNet":
        self._prepare(history)
        for horizon in self.horizons:
            train, y, prediction_rows = self._horizon_rows(horizon)
            if train.empty:
                raise ValueError(f"no training rows remain for horizon {horizon}")
            matrix = self._matrix(train)
            self._n_continuous = matrix.shape[1] - (len(self._series_ids) - 1) - 12
            apply_matrix = self._matrix(self._prediction_rows())
            mean = matrix[:, : self._n_continuous].mean(axis=0)
            std = matrix[:, : self._n_continuous].std(axis=0)
            std_safe = np.where(std > 0, std, 1.0)
            matrix[:, : self._n_continuous] = (matrix[:, : self._n_continuous] - mean) / std_safe
            apply_matrix = apply_matrix.copy()
            apply_matrix[:, : self._n_continuous] = (
                apply_matrix[:, : self._n_continuous] - mean
            ) / std_safe
            alpha, l1_ratio = self._select_regularization(matrix, y, train)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
            self._models[horizon] = model.fit(matrix, y)
            self._apply_matrices[horizon] = apply_matrix
        return self

    def _select_regularization(self, matrix: np.ndarray, y: np.ndarray, train: pd.DataFrame) -> tuple[float, float]:
        """Inner expanding-window CV on training rows only (predeclared §6)."""
        months = train["observation_time"].dt.to_period("M")
        unique_months = np.sort(months.unique())
        if len(unique_months) <= ELASTIC_INNER_VAL_MONTHS * (ELASTIC_INNER_FOLDS + 1):
            return ELASTIC_DEFAULT
        month_pos = months.map({period: pos for pos, period in enumerate(unique_months)}).to_numpy()
        n_series = len(self._series_ids)
        errors: dict[tuple[float, float], list[float]] = {}
        folds_used = 0
        expected_per_fold = ELASTIC_INNER_VAL_MONTHS * n_series
        for fold in range(ELASTIC_INNER_FOLDS):
            val_start = len(unique_months) - ELASTIC_INNER_VAL_MONTHS * (fold + 1)
            val_mask = (month_pos >= val_start) & (
                month_pos < val_start + ELASTIC_INNER_VAL_MONTHS
            )
            train_mask = month_pos < val_start
            if not train_mask.any():
                continue
            if len(np.unique(month_pos[train_mask])) < ELASTIC_MIN_TRAIN_MONTHS:
                continue
            folds_used += 1
            train_mean = matrix[train_mask, : self._n_continuous].mean(axis=0)
            train_std = matrix[train_mask, : self._n_continuous].std(axis=0)
            std_safe = np.where(train_std > 0, train_std, 1.0)
            fold_train = matrix[train_mask].copy()
            fold_val = matrix[val_mask].copy()
            fold_train[:, : self._n_continuous] = (
                fold_train[:, : self._n_continuous] - train_mean
            ) / std_safe
            fold_val[:, : self._n_continuous] = (
                fold_val[:, : self._n_continuous] - train_mean
            ) / std_safe
            for alpha in ELASTIC_ALPHAS:
                for l1_ratio in ELASTIC_L1_RATIOS:
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
                    model.fit(fold_train, y[train_mask])
                    residuals = np.abs(y[val_mask] - model.predict(fold_val))
                    errors.setdefault((alpha, l1_ratio), []).extend(residuals.tolist())
        if folds_used == 0:
            return ELASTIC_DEFAULT
        scored = {
            config: float(np.mean(residuals))
            for config, residuals in errors.items()
            if len(residuals) == folds_used * expected_per_fold
        }
        if not scored:
            return ELASTIC_DEFAULT
        best_error = min(scored.values())
        best_configs = [config for config, error in scored.items() if error <= best_error]
        return min(best_configs)  # smallest alpha, then smallest l1_ratio

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("fit must be called before predict")
        if horizon != max(self.horizons):
            raise ValueError("direct models are fit for the full horizon set")

        def per_horizon_predictions(h: int) -> np.ndarray:
            return self._models[h].predict(self._apply_matrices[h])

        return self._predictions(per_horizon_predictions)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {
                "alphas": list(ELASTIC_ALPHAS),
                "l1_ratios": list(ELASTIC_L1_RATIOS),
                "inner_val_months": ELASTIC_INNER_VAL_MONTHS,
                "inner_folds": ELASTIC_INNER_FOLDS,
                "min_train_months": ELASTIC_MIN_TRAIN_MONTHS,
                "default": list(ELASTIC_DEFAULT),
            },
            "groups": list(self.groups),
            "features": [c for g in self.groups for c in GROUP_FEATURES[g]],
            "series_ids": list(self._series_ids),
        }
