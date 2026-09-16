"""Mask numeric training cells by replacing them with the column mean.

Replaces 5% of numeric training cells with the per-column training mean.
Unlike NaN dropout, this drives the model toward the column's central
tendency rather than the missing-value branch, which can act as a mild
shrinkage regularizer on individual splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MASK_RATE = 0.05


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def mean_imputation_dropout_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    num_cols = _numeric_cols(out)
    if not num_cols:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    for col in num_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        mean = float(np.mean(finite))
        if not np.isfinite(mean):
            continue
        mask = rng.random(n) < _MASK_RATE
        if not mask.any():
            continue
        vals = vals.copy()
        vals[mask] = np.float32(mean)
        out[col] = vals
    return out, np.asarray(y_train).copy()
