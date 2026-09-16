"""Randomly mask a small fraction of categorical training cells.

Sets 5% of categorical/object cells in the training data to the string
``"__nan__"``, which the pipeline's frame preparation routes into the
shared missing-value branch. Forces LightGBM to find robust splits that
don't depend on a specific categorical level always being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DROP_RATE = 0.05


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        dtype = df[c].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cols.append(c)
    return cols


def cat_dropout_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    cat_cols = _categorical_cols(out)
    if not cat_cols:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    for col in cat_cols:
        mask = rng.random(n) < _DROP_RATE
        if not mask.any():
            continue
        series = out[col].astype("string").fillna("__nan__").to_numpy()
        series = series.copy()
        series[mask] = "__nan__"
        out[col] = pd.Categorical(series)
    return out, np.asarray(y_train).copy()
