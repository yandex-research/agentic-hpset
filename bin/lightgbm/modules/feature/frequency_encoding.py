"""Train-frequency encoding of categorical columns.

For each categorical column, replaces each level with the count of that
level observed in train. Unseen levels in val/test get a count of zero.
This exposes popularity-of-level signal as a numeric feature without
inflating dimensionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREFIX = "freq"


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


def frequency_encoding_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if not cat_cols:
        return out
    for col in cat_cols:
        train_series = x["train"][col].astype("string").fillna("__nan__")
        counts = train_series.value_counts()
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            series = frame[col].astype("string").fillna("__nan__")
            values = series.map(counts).fillna(0.0).astype(np.float32).to_numpy()
            out[part][new_name] = values
    return out
