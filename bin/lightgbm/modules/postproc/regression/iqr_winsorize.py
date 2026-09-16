"""IQR-based winsorization of regression predictions.

Clips predictions to `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` computed from
`y_train`, the standard Tukey fence. More tolerant than the [min, max]
clip because it ignores extreme training outliers, and more aggressive
than the [1%, 99%] quantile clip when the target distribution has
moderate spread.
"""

from __future__ import annotations

import numpy as np

_K = 1.5


def iqr_winsorize_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {part: np.asarray(arr).copy() for part, arr in preds.items()}
    q1 = float(np.quantile(finite, 0.25))
    q3 = float(np.quantile(finite, 0.75))
    iqr = q3 - q1
    lo, hi = q1 - _K * iqr, q3 + _K * iqr
    if hi < lo:
        lo, hi = hi, lo
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        out[part] = np.clip(np.asarray(arr, dtype=np.float64), lo, hi).astype(np.float32)
    return out
