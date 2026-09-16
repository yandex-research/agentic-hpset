"""Smooth probabilities with a fixed exponent and renormalize.

Applies `p ← p ** 0.9` and renormalizes each row. Pulls probabilities
toward the uniform distribution — opposite of `power_sharpen` — which
tends to improve log-loss on overconfident models. Fixed exponent so
there is no validation tuning involved.
"""

from __future__ import annotations

import numpy as np

_GAMMA = 0.9
_EPS = 1e-12


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def power_smooth_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        smooth = np.power(np.maximum(a, _EPS), _GAMMA)
        row_sum = smooth.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (smooth / row_sum).astype(np.float32)
    return out
