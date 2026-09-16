"""Row-wise summary statistics across numeric features.

Adds mean, std, min, max, and median computed across each row's numeric
columns. Trees cannot easily compute these row-level aggregates from
column-wise splits, so handing them over as explicit columns can expose
useful signal at constant per-row cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def row_stats_features(
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
) -> dict[str, pd.DataFrame]:
    out = {part: frame.copy() for part, frame in x.items()}
    num_cols = _numeric_cols(x["train"])
    if not num_cols:
        return out
    for part, frame in out.items():
        block = frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            row_mean = np.nanmean(block, axis=1)
            row_std = np.nanstd(block, axis=1)
            row_min = np.nanmin(block, axis=1)
            row_max = np.nanmax(block, axis=1)
            row_median = np.nanmedian(block, axis=1)
        out[part]["rowstat_mean"] = row_mean.astype(np.float32)
        out[part]["rowstat_std"] = row_std.astype(np.float32)
        out[part]["rowstat_min"] = row_min.astype(np.float32)
        out[part]["rowstat_max"] = row_max.astype(np.float32)
        out[part]["rowstat_median"] = row_median.astype(np.float32)
    return out
