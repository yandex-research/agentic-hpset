"""Clip regression predictions at zero when the train target is non-negative.

If every finite `y_train` value is non-negative, predictions are clipped
to `[0, +inf)`. Otherwise the predictions pass through unchanged. Useful
for targets like prices, counts, or durations where LightGBM can produce
small negative outputs at the tails.
"""

from __future__ import annotations

import numpy as np


def nonneg_clip_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    is_nonneg = bool(finite.size) and bool(np.all(finite >= 0.0))
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        vals = np.asarray(arr, dtype=np.float64)
        if is_nonneg:
            vals = np.maximum(vals, 0.0)
        out[part] = vals.astype(np.float32)
    return out
