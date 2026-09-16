"""Quantile-match regression predictions to the training-target CDF.

Each prediction is replaced by the training-target value at the same
empirical rank: `pred_rank(x) -> y_train_sorted[round(rank * (n - 1))]`.
Uses only `y_train` statistics, so no validation overfit. This forces
predictions to share the training-target marginal, which can help when
the model's raw output is systematically compressed.
"""

from __future__ import annotations

import numpy as np


def quantile_match_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    sorted_y = np.sort(finite)
    n = sorted_y.size
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        vals = np.asarray(arr, dtype=np.float64)
        ranks = np.argsort(np.argsort(vals)).astype(np.float64)
        if vals.size > 1:
            ranks /= vals.size - 1
        else:
            ranks = np.zeros_like(ranks)
        idx = np.clip(np.round(ranks * (n - 1)).astype(np.int64), 0, n - 1)
        out[part] = sorted_y[idx].astype(np.float32)
    return out
