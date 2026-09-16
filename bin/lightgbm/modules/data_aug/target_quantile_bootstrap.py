"""Bootstrap weighted toward extreme-target rows.

Buckets training rows into 10 target deciles and resamples (with
replacement) so that the tail deciles are 50% more likely than the
central deciles. Designed for regression where the model is often
under-trained on the tails of the target distribution. For classification
the heuristic still upweights minority classes if `y_train` has at most
10 unique values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_N_BINS = 10
_TAIL_WEIGHT = 1.5


def target_quantile_bootstrap_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x_train)
    y = np.asarray(y_train, dtype=np.float64)
    if n == 0:
        return x_train.copy(), np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    finite = y[np.isfinite(y)]
    if finite.size < 2:
        idx = rng.integers(0, n, size=n)
        return x_train.iloc[idx].reset_index(drop=True), y[idx].copy()
    uniq = np.unique(finite)
    if uniq.size <= _N_BINS:
        codes = pd.Categorical(y, categories=uniq).codes.copy()
        codes[codes < 0] = uniq.size
        bins = codes.astype(np.int32)
        n_bins = uniq.size + (1 if (codes == uniq.size).any() else 0)
    else:
        qs = np.linspace(0.0, 1.0, _N_BINS + 1)
        edges = np.unique(np.quantile(finite, qs))
        if edges.size < 2:
            idx = rng.integers(0, n, size=n)
            return x_train.iloc[idx].reset_index(drop=True), y[idx].copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        bins = np.digitize(y, edges[1:-1], right=False).astype(np.int32)
        n_bins = edges.size - 1
    weights = np.ones(n, dtype=np.float64)
    centers = (n_bins - 1) / 2.0
    for b in range(n_bins):
        dist = abs(b - centers) / max(centers, 1.0)
        boost = 1.0 + (_TAIL_WEIGHT - 1.0) * dist
        mask = bins == b
        if mask.any():
            weights[mask] = boost
    probs = weights / weights.sum()
    idx = rng.choice(n, size=n, replace=True, p=probs)
    out_x = x_train.iloc[idx].reset_index(drop=True)
    out_y = np.asarray(y_train)[idx].copy()
    return out_x, out_y
