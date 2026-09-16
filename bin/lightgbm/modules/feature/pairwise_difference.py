"""Pairwise differences of top-variance numeric features.

Picks up to 8 numeric columns with the largest train variance and adds all
pairwise signed differences `a - b` as new numeric columns. Bounded by
C(K,2)=28 extra columns. Differences expose comparisons between paired
measurements that single splits cannot express directly.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

_TOP_K = 8
_PREFIX = "diff"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def pairwise_difference_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if len(num_cols) < 2:
        return out
    train = x["train"]
    variances = {
        c: float(
            np.nanvar(pd.to_numeric(train[c], errors="coerce").to_numpy(np.float64))
        )
        for c in num_cols
    }
    chosen = sorted(num_cols, key=lambda c: -variances[c])[:_TOP_K]
    if len(chosen) < 2:
        return out
    for a, b in combinations(chosen, 2):
        new_name = f"{_PREFIX}_{a}_m_{b}"
        for part, frame in out.items():
            va = pd.to_numeric(frame[a], errors="coerce").to_numpy(np.float32)
            vb = pd.to_numeric(frame[b], errors="coerce").to_numpy(np.float32)
            out[part][new_name] = (va - vb).astype(np.float32)
    return out
