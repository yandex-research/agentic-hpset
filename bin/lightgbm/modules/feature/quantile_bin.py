"""Quantile-based binning of numeric features.

For each numeric column, compute 10 quantile bins on the training data and
apply the bin index to every split. The bin index is exposed as a new
categorical column so LightGBM can model non-monotonic structure that hides
behind raw numeric ordering. Cost is linear in rows times numeric columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_N_BINS = 10
_PREFIX = "qbin"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def quantile_bin_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out
    qs = np.linspace(0.0, 1.0, _N_BINS + 1)
    for col in num_cols:
        train_vals = pd.to_numeric(x["train"][col], errors="coerce").to_numpy(np.float64)
        finite = train_vals[np.isfinite(train_vals)]
        if finite.size == 0:
            continue
        edges = np.unique(np.quantile(finite, qs))
        if edges.size < 2:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
            bins = np.digitize(vals, edges[1:-1], right=False).astype(np.int32)
            out[part][new_name] = pd.Categorical(bins)
    return out
