"""Swap-noise augmentation across all training columns.

For each column, randomly replaces a small fraction (3%) of rows with the
value at another randomly chosen training row in the same column. Labels
are preserved. The noise is realistic (it preserves marginals) so it
typically degrades performance less than Gaussian noise on heterogeneous
tabular data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SWAP_RATE = 0.03


def column_swap_noise_aug(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = x_train.copy()
    n = len(out)
    if n < 2:
        return out, np.asarray(y_train).copy()
    rng = np.random.default_rng(seed)
    for col in out.columns:
        mask = rng.random(n) < _SWAP_RATE
        if not mask.any():
            continue
        replace_with = rng.integers(0, n, size=int(mask.sum()))
        out_col = out[col].to_numpy(copy=True)
        out_col[mask] = out[col].to_numpy()[replace_with]
        out[col] = out_col
    return out, np.asarray(y_train).copy()
