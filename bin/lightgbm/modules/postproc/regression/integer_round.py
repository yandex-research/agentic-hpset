"""Round regression predictions to integers when the train target is integer-valued.

If every finite `y_train` value is integer (within 1e-6), predictions are
rounded to the nearest integer (still returned as float32 so downstream
metrics behave the same). Otherwise the predictions pass through unchanged.
Useful for count-style targets that LightGBM models as real-valued.
"""

from __future__ import annotations

import numpy as np


def integer_round_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    is_int = bool(finite.size) and bool(np.allclose(finite, np.round(finite), atol=1e-6))
    out: dict[str, np.ndarray] = {}
    for part, arr in preds.items():
        vals = np.asarray(arr, dtype=np.float64)
        if is_int:
            vals = np.round(vals)
        out[part] = vals.astype(np.float32)
    return out
