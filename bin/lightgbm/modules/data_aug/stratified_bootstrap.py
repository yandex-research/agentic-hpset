"""Bootstrap that resamples within target strata.

Buckets `y_train` into up to 10 strata (by unique value for small label
sets, by decile otherwise) and resamples within each stratum with
replacement. The total row count is preserved. Compared to a plain
bootstrap this keeps the target-marginal distribution stable, which can
matter for highly imbalanced classification or skewed regression targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAX_STRATA = 10


def _make_strata(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y[~np.isnan(y)] if y.dtype.kind == "f" else y)
    if uniq.size <= _MAX_STRATA:
        codes = pd.Categorical(y, categories=uniq).codes.copy()
        codes[codes < 0] = uniq.size
        return codes.astype(np.int32)
    qs = np.linspace(0.0, 1.0, _MAX_STRATA + 1)
    edges = np.unique(np.quantile(y[np.isfinite(y)], qs))
    if edges.size < 2:
        return np.zeros(y.shape[0], dtype=np.int32)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(y, edges[1:-1], right=False).astype(np.int32)


def stratified_bootstrap_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x_train)
    if n == 0:
        return x_train.copy(), np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train)
    strata = _make_strata(y)
    selected = np.empty(n, dtype=np.int64)
    pos = 0
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if idx.size == 0:
            continue
        picks = rng.integers(0, idx.size, size=idx.size)
        selected[pos:pos + idx.size] = idx[picks]
        pos += idx.size
    selected = selected[:pos]
    if selected.size == 0:
        return x_train.copy(), np.asarray(y_train).copy()
    out_x = x_train.iloc[selected].reset_index(drop=True)
    out_y = y[selected].copy()
    return out_x, out_y
