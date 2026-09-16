"""Smooth tanh-based clipping to the training-target range.

Maps each prediction `p` through a tanh squashing centered on the train
mean with width `(max - min) / 2`. Unlike hard clipping, predictions
near the boundary are smoothly attenuated rather than jumping at a wall,
which can be preferable for metrics like RMSE that penalize large
boundary residuals more than near-edge ones.
"""

from __future__ import annotations

import numpy as np


def softclip_tanh_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        vals = np.asarray(arr, dtype=np.float64)
        squashed = center + half_width * np.tanh((vals - center) / max(half_width, 1e-9))
        out[part] = squashed.astype(np.float32)
    return out
