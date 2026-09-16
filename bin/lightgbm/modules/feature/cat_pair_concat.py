"""Concatenated key from the two lowest-cardinality categorical columns.

Picks two categorical columns with the smallest train cardinality and adds
a new categorical column whose values are the joined string of their two
levels. This gives LightGBM an explicit interaction key without exploding
cardinality the way a one-hot crossing would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREFIX = "catpair"
_MAX_CARD = 256


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


def cat_pair_concat_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if len(cat_cols) < 2:
        return out
    cardinalities = {
        c: int(x["train"][c].astype("string").nunique(dropna=False)) for c in cat_cols
    }
    eligible = [c for c, k in cardinalities.items() if k <= _MAX_CARD]
    if len(eligible) < 2:
        return out
    a, b = sorted(eligible, key=lambda c: cardinalities[c])[:2]
    new_name = f"{_PREFIX}_{a}_{b}"
    for part, frame in out.items():
        sa = frame[a].astype("string").fillna("__nan__")
        sb = frame[b].astype("string").fillna("__nan__")
        joined = (sa + "||" + sb).to_numpy()
        out[part][new_name] = pd.Categorical(joined)
    return out
