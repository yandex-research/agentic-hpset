"""Small shrinkage of regression predictions toward the train mean.

Returns `0.98 * preds + 0.02 * mean(y_train)`. The small mixing weight
acts as a James-Stein-style bias-variance trade — it nudges predictions
slightly toward a stable training-only anchor, which can help on noisy
or small validation sets without ever fitting to val data.
"""

from __future__ import annotations

import numpy as np

_SHRINK = 0.02


def shrink_to_train_mean_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    mean = float(np.mean(finite))
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        vals = np.asarray(arr, dtype=np.float64)
        shrunk = (1.0 - _SHRINK) * vals + _SHRINK * mean
        out[part] = shrunk.astype(np.float32)
    return out
