"""Regime-switch policy layer for the MRTS vintage satellite.

Builds and evaluates the predeclared policy of POLICY.md: a small causal
classifier that picks naive or theta per (category, forecast origin).
The policy menu is exactly {naive, theta}; labels come from the frozen
fold files; the 2023+ holdout is never read.
"""

from regime.causality import (
    assert_error_lag,
    assert_origin_window,
    assert_snapshot_causal,
    assert_train_before_eval,
    error_publish_time,
    errors_published,
)
from regime.labels import FoldData, LabelTable, build_labels, load_fold_data
from regime.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_feature_table

__all__ = [
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "FoldData",
    "LabelTable",
    "assert_error_lag",
    "assert_origin_window",
    "assert_snapshot_causal",
    "assert_train_before_eval",
    "build_feature_table",
    "build_labels",
    "error_publish_time",
    "errors_published",
    "load_fold_data",
]
