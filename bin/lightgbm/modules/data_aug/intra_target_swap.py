"""Swap a subset of feature columns with a same-target neighbor.

For 10% of training rows, picks a random partner row that shares the
target value (or target decile for regression) and copies the partner's
values over a random ~30% subset of columns. Labels stay the same. This
amounts to a same-class CutMix variant — it injects realistic within-
class variability without touching label semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ROW_FRACTION = 0.10
_COL_FRACTION = 0.30
_MAX_BINS = 10


def _target_buckets(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=np.int32)
    uniq = np.unique(y[np.isfinite(y)])
    if uniq.size <= _MAX_BINS:
        codes = pd.Categorical(y, categories=uniq).codes.copy()
        codes[codes < 0] = uniq.size
        return codes.astype(np.int32)
    qs = np.linspace(0.0, 1.0, _MAX_BINS + 1)
    edges = np.unique(np.quantile(y[np.isfinite(y)], qs))
    if edges.size < 2:
        return np.zeros(y.shape[0], dtype=np.int32)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(y, edges[1:-1], right=False).astype(np.int32)


def intra_target_swap_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    n = len(out)
    if n < 2 or out.shape[1] == 0:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train)
    buckets = _target_buckets(y)
    swap_mask = rng.random(n) < _ROW_FRACTION
    swap_rows = np.flatnonzero(swap_mask)
    if swap_rows.size == 0:
        return out, y.copy()
    bucket_to_rows: dict[int, np.ndarray] = {}
    for b in np.unique(buckets):
        bucket_to_rows[int(b)] = np.flatnonzero(buckets == b)
    n_cols_to_swap = max(1, int(round(_COL_FRACTION * out.shape[1])))
    all_cols = list(out.columns)
    new_vals: dict[str, np.ndarray] = {}
    for col in all_cols:
        new_vals[col] = out[col].to_numpy(copy=True)
    for row in swap_rows:
        pool = bucket_to_rows.get(int(buckets[row]))
        if pool is None or pool.size < 1:
            continue
        partner = int(pool[rng.integers(0, pool.size)])
        if partner == row:
            continue
        swap_cols = rng.choice(out.shape[1], size=n_cols_to_swap, replace=False)
        for ci in swap_cols:
            col = all_cols[int(ci)]
            new_vals[col][row] = out[col].to_numpy()[partner]
    for col in all_cols:
        out[col] = new_vals[col]
    return out, y.copy()
