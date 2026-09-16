"""Clip regression predictions to the [1%, 99%] training-target quantile.

Looser than the hard min/max clip — predictions are bounded to the
inner 98% range of the training target. Avoids the failure mode where
a single outlier in `y_train` makes the hard-range clip a no-op while
still trimming wild predictions.
"""

from __future__ import annotations

import numpy as np

_LO_Q = 0.01
_HI_Q = 0.99


def clip_train_quantile_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    lo, hi = float(np.quantile(finite, _LO_Q)), float(np.quantile(finite, _HI_Q))
    if hi < lo:
        lo, hi = hi, lo
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        out[part] = np.clip(np.asarray(arr, dtype=np.float64), lo, hi).astype(np.float32)
    return out
