"""Point forecast metrics — port of retailPRED ``forecasting/evaluation/metrics.py``.

Only the subset used by the vintage gate (wape/mae/bias) is ported; behavior
is identical.
"""

from __future__ import annotations

import numpy as np


def _arrays(actual: object, prediction: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(prediction, dtype=float)
    if y.shape != y_hat.shape or y.size == 0:
        raise ValueError("actual and prediction must have the same non-empty shape")
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        raise ValueError("metrics require finite values")
    return y, y_hat


def mae(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    return float(np.mean(np.abs(y - y_hat)))


def wape(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    denominator = np.sum(np.abs(y))
    if denominator == 0:
        raise ValueError("WAPE is undefined when all actuals are zero")
    return float(np.sum(np.abs(y - y_hat)) / denominator)


def bias(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    return float(np.mean(y_hat - y))
