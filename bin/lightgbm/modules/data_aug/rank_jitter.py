"""Jitter each numeric column in its empirical rank space.

For each numeric column, computes ranks on the training data, perturbs
the ranks by a small amount, and maps them back to values using the
sorted training array. Preserves the column's marginal distribution
while breaking exact-tie patterns; designed to be gentler than additive
noise on heavy-tailed columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_JITTER_FRACTION = 0.02


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def rank_jitter_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    num_cols = _numeric_cols(out)
    if not num_cols:
        return out, np.asarray(y_train).copy()
    n = len(out)
    if n < 2:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    shift = max(1, int(round(_JITTER_FRACTION * n)))
    for col in num_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float64)
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() < 2:
            continue
        finite_vals = vals[finite_mask]
        order = np.argsort(finite_vals, kind="mergesort")
        ranks = np.empty(finite_vals.size, dtype=np.int64)
        ranks[order] = np.arange(finite_vals.size, dtype=np.int64)
        delta = rng.integers(-shift, shift + 1, size=finite_vals.size)
        new_ranks = np.clip(ranks + delta, 0, finite_vals.size - 1)
        sorted_vals = np.sort(finite_vals)
        new_finite = sorted_vals[new_ranks]
        new_vals = vals.copy()
        new_vals[finite_mask] = new_finite
        out[col] = new_vals.astype(np.float32)
    return out, np.asarray(y_train).copy()
