from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import (
    column_centers,
    fill_nans,
    filter_constant_columns,
    signed_log1p,
)


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Apply signed log1p, then standardize each column.

    Hypothesis: many tabular features (income, area, counts) live on multiplicative
    scales with right-skew. Signed log1p compresses the tail without imposing the
    PLR's full empirical-quantile structure, giving the model linear-on-log inputs
    that align with the way tree models split on logarithmic axes.
    """
    if x_num is None:
        return None, {"mean": None, "std": None, "keep_mask": None}
    config = config or {}
    # Clip inputs to the train percentile range before log1p + standardization
    # so an out-of-range test value cannot be amplified into a catastrophic
    # network input. Mirrors the guard in clipped_quantile (v6).
    x_num_train = np.asarray(x_num["train"], dtype=np.float64)
    lo = np.nanpercentile(x_num_train, float(config.get("clip_low_pct", 1.0)), axis=0)
    hi = np.nanpercentile(x_num_train, float(config.get("clip_high_pct", 99.0)), axis=0)
    same = hi <= lo
    if np.any(same):
        lo = np.where(same, np.nanmin(x_num_train, axis=0), lo)
        hi = np.where(same, np.nanmax(x_num_train, axis=0), hi)
    train_log = signed_log1p(np.clip(x_num_train, lo, hi))
    mean = column_centers(train_log, np.nanmean)
    train_filled = fill_nans(train_log, mean)
    std = train_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
        arr = fill_nans(signed_log1p(clipped), mean)
        transformed[part] = ((arr - mean) / std).astype(np.float32)
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {"mean": mean, "std": std, "keep_mask": keep_mask}
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
