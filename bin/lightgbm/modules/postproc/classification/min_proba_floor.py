"""Apply a minimum probability floor and renormalize rows.

Replaces any probability below 1e-3 with 1e-3 and renormalizes each
row. Heavier than the standard `clip_renormalize` floor (1e-6); useful
when log-loss is dominated by a handful of extremely confident wrong
predictions, while leaving the argmax-driven accuracy mostly untouched.
"""

from __future__ import annotations

import numpy as np

_FLOOR = 1e-3


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def min_proba_floor_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        floored = np.maximum(a, _FLOOR)
        row_sum = floored.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out[part] = (floored / row_sum).astype(np.float32)
    return out
