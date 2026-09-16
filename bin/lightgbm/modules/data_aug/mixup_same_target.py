"""Within-target mixup of numeric columns only.

For 20% of training rows, replace each numeric column with a convex
combination `lam * x_i + (1 - lam) * x_j` where `x_j` is a random row
with the same target value (or the same target decile for regression).
`lam ~ Beta(0.4, 0.4)` is chosen per row. Categorical columns and the
label stay untouched so this is safe for both task types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIX_FRACTION = 0.2
_BETA_ALPHA = 0.4
_MAX_BINS = 10


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


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


def mixup_same_target_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    n = len(out)
    if n < 2:
        return out, np.asarray(y_train).copy()
    num_cols = _numeric_cols(out)
    if not num_cols:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train)
    buckets = _target_buckets(y)
    mix_mask = rng.random(n) < _MIX_FRACTION
    mix_rows = np.flatnonzero(mix_mask)
    if mix_rows.size == 0:
        return out, y.copy()
    bucket_to_rows: dict[int, np.ndarray] = {}
    for b in np.unique(buckets):
        bucket_to_rows[int(b)] = np.flatnonzero(buckets == b)
    partners = np.empty(mix_rows.size, dtype=np.int64)
    for k, row in enumerate(mix_rows):
        pool = bucket_to_rows.get(int(buckets[row]))
        if pool is None or pool.size < 1:
            partners[k] = row
            continue
        partners[k] = pool[rng.integers(0, pool.size)]
    lam = rng.beta(_BETA_ALPHA, _BETA_ALPHA, size=mix_rows.size).astype(np.float32)
    for col in num_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(np.float32)
        new_vals = vals.copy()
        a = vals[mix_rows]
        b = vals[partners]
        with np.errstate(invalid="ignore"):
            mixed = lam * a + (1.0 - lam) * b
            mixed = np.where(np.isnan(mixed), a, mixed)
        new_vals[mix_rows] = mixed.astype(np.float32)
        out[col] = new_vals
    return out, y.copy()
