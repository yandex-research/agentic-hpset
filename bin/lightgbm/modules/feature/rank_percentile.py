"""Train-fit rank-percentile transform of numeric columns.

For each numeric column, computes a percentile transform fit on train: each
value is mapped to its empirical CDF in [0, 1]. Replaces the numeric column
in place. Trees are invariant to this monotone transform when fit alone,
but the rescaled basis can interact differently with other engineered
features under append mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREFIX = "rank"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def rank_percentile_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out
    for col in num_cols:
        train_vals = pd.to_numeric(x["train"][col], errors="coerce").to_numpy(np.float64)
        finite = train_vals[np.isfinite(train_vals)]
        if finite.size == 0:
            continue
        sorted_train = np.sort(finite)
        n_finite = sorted_train.size
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
            ranks = np.searchsorted(sorted_train, vals, side="right").astype(np.float64)
            pct = ranks / max(n_finite, 1)
            pct = np.where(np.isnan(vals), np.nan, pct)
            out[part][new_name] = pct.astype(np.float32)
    return out
