"""Per-column NaN indicators plus a row-level NaN count.

Adds one binary indicator per numeric column that has any NaNs in train, and
a single `nan_count` column that counts missing values per row. LightGBM
already handles NaN by routing, but explicit missingness columns let it
split on missingness in conjunction with other features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREFIX = "nanind"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def nan_indicator_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out
    train_nan_cols = [
        c for c in num_cols
        if pd.to_numeric(x["train"][c], errors="coerce").isna().any()
    ]
    for part, frame in out.items():
        block = frame[num_cols].apply(pd.to_numeric, errors="coerce")
        out[part]["nan_count"] = block.isna().sum(axis=1).astype(np.float32).to_numpy()
        for col in train_nan_cols:
            out[part][f"{_PREFIX}_{col}"] = (
                block[col].isna().astype(np.float32).to_numpy()
            )
    return out
