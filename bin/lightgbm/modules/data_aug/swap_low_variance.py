"""Swap-noise restricted to the lowest-variance numeric columns.

Replaces 10% of cells in the bottom-half (by train variance) numeric
columns with values drawn from another random row. Low-variance columns
are common targets for spurious splits in noisy data; this perturbation
specifically nudges the model away from over-using them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SWAP_RATE = 0.10


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def swap_low_variance_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    num_cols = _numeric_cols(out)
    n = len(out)
    if len(num_cols) < 2 or n < 2:
        return out, np.asarray(y_train).copy()
    variances = {}
    for c in num_cols:
        vals = pd.to_numeric(out[c], errors="coerce").to_numpy(np.float64)
        finite = vals[np.isfinite(vals)]
        variances[c] = float(np.var(finite)) if finite.size else 0.0
    sorted_cols = sorted(num_cols, key=lambda c: variances[c])
    target_cols = sorted_cols[: max(1, len(sorted_cols) // 2)]
    rng = np.random.default_rng(seed)
    for col in target_cols:
        mask = rng.random(n) < _SWAP_RATE
        if not mask.any():
            continue
        replace_with = rng.integers(0, n, size=int(mask.sum()))
        out_col = out[col].to_numpy(copy=True)
        out_col[mask] = out[col].to_numpy()[replace_with]
        out[col] = out_col
    return out, np.asarray(y_train).copy()
