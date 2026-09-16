"""Clip per-class probabilities and renormalize each row to sum to 1.

Each probability is clipped to `[1e-6, 1 - 1e-6]` and rows are
renormalized. Removes the pathological zeros that LightGBM occasionally
emits at extreme leaves, which is helpful for log-loss-style metrics
where a true zero is unbounded penalty. Uses no validation information.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def clip_renormalize_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        a = np.clip(a, _EPS, 1.0 - _EPS)
        row_sum = a.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (a / row_sum).astype(np.float32)
    return out
