"""Collapse rare categorical levels into a shared `__rare__` token.

For each categorical column, levels whose train frequency is below 1%
(with a minimum count of 5) are remapped to `__rare__`. Reduces split
fragmentation on high-cardinality columns and consolidates noisy levels
that the model can't reliably learn from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_FREQ = 0.01
_MIN_COUNT = 5
_RARE_TOKEN = "__rare__"


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


def rare_category_collapse_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if not cat_cols:
        return out
    n = len(x["train"])
    min_count = max(_MIN_COUNT, int(np.ceil(_MIN_FREQ * n)))
    for col in cat_cols:
        train_series = x["train"][col].astype("string").fillna("__nan__")
        counts = train_series.value_counts()
        keep = set(counts.index[counts >= min_count].tolist())
        if not keep:
            continue
        for part, frame in out.items():
            series = frame[col].astype("string").fillna("__nan__")
            collapsed = series.where(series.isin(keep), _RARE_TOKEN).to_numpy()
            out[part][col] = pd.Categorical(collapsed)
    return out
