"""Clip regression predictions to the training-target value range.

Hard-caps `val` and `test` predictions at `[min(y_train), max(y_train)]`.
Defends against extrapolation that LightGBM cannot do by construction
(constant leaves) without using validation information, so this carries
no validation-overfitting risk.
"""

from __future__ import annotations

import numpy as np


def clip_train_range_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    lo, hi = float(finite.min()), float(finite.max())
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        out[part] = np.clip(np.asarray(arr, dtype=np.float64), lo, hi).astype(np.float32)
    return out
