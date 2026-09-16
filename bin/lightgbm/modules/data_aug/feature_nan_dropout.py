"""Randomly mask a small fraction of numeric training cells as NaN.

Sets 5% of numeric cells in the training data to NaN, chosen i.i.d. per
cell. LightGBM treats missing values as a separate split branch, so this
forces the model to learn alternative routing paths and discourages
reliance on any single feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DROP_RATE = 0.05


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def feature_nan_dropout_aug(
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
        mask = rng.random(n) < _DROP_RATE
        if not mask.any():
            continue
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float32)
        vals = vals.copy()
        vals[mask] = np.nan
        out[col] = vals
    return out, np.asarray(y_train).copy()
