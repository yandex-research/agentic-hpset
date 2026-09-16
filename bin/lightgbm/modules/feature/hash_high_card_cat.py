"""Hash high-cardinality categorical columns into a small bucket set.

For each categorical column with train cardinality above 64, the level
string is hashed into 32 buckets. The bucketed version is added as a new
categorical column (the original is kept untouched). Bounds the per-split
cardinality LightGBM has to consider for sparse categoricals.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_CARD_THRESHOLD = 64
_N_BUCKETS = 32
_PREFIX = "hashcat"


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


def _hash_token(s: str) -> int:
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _N_BUCKETS


def hash_high_card_cat_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    cat_cols = _categorical_cols(x["train"])
    if not cat_cols:
        return out
    for col in cat_cols:
        card = int(x["train"][col].astype("string").nunique(dropna=False))
        if card <= _CARD_THRESHOLD:
            continue
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            series = frame[col].astype("string").fillna("__nan__")
            buckets = series.map(_hash_token).astype(np.int32).to_numpy()
            out[part][new_name] = pd.Categorical(buckets)
    return out
