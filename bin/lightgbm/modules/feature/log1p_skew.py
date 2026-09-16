"""log1p of right-skewed non-negative numeric columns.

For each numeric column whose train values are non-negative and have skew
above a threshold, adds `log1p(x)` as a new column. LightGBM is invariant
to monotone transforms when fit by itself, but the log version supplies a
different basis for interactions with other features in `append` mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SKEW_THRESHOLD = 1.0
_PREFIX = "log1p"


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def log1p_skew_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out
    train = x["train"]
    for col in num_cols:
        vals = pd.to_numeric(train[col], errors="coerce").to_numpy(np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        if finite.min() < 0.0:
            continue
        s = pd.Series(finite)
        try:
            skew = float(s.skew())
        except Exception:
            continue
        if not np.isfinite(skew) or abs(skew) < _SKEW_THRESHOLD:
            continue
        new_name = f"{_PREFIX}_{col}"
        for part, frame in out.items():
            arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(np.float64)
            arr = np.where(arr < 0.0, 0.0, arr)
            out[part][new_name] = np.log1p(arr).astype(np.float32)
    return out
