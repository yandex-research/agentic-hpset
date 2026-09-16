"""Label smoothing toward the uniform distribution.

Blends each row with the uniform distribution: `p' = (1 - eps) * p +
eps * uniform`, with `eps = 0.02`. Mild regularizer that prevents
overconfident probabilities and improves log-loss when the model is
slightly miscalibrated, without using any validation information.
"""

from __future__ import annotations

import numpy as np

_EPS = 0.02


def _binarize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.column_stack([1.0 - arr, arr])
    return arr


def label_smoothing_postproc(
    proba: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for part, arr in proba.items():
        a = _binarize(np.asarray(arr, dtype=np.float64))
        k = a.shape[1]
        if k == 0:
            out[part] = a.astype(np.float32)
            continue
        smoothed = (1.0 - _EPS) * a + _EPS / k
        out[part] = smoothed.astype(np.float32)
    return out
