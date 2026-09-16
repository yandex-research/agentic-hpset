"""Sharpen probabilities with a fixed exponent and renormalize.

Applies `p ← p ** 1.1` and renormalizes each row to sum to 1. Pushes
probabilities toward 0 or 1, which can lift accuracy on argmax-decided
classification while slightly worsening calibration. Fixed exponent so
there is no validation tuning involved.
"""

from __future__ import annotations

import numpy as np

_GAMMA = 1.1
_EPS = 1e-12


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def power_sharpen_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        sharp = np.power(np.maximum(a, _EPS), _GAMMA)
        row_sum = sharp.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (sharp / row_sum).astype(np.float32)
    return out
