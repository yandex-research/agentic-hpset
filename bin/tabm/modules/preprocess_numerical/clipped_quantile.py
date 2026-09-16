from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import (
    filter_constant_columns,
    fit_quantile_normal_transformer,
)


def numerical_preprocess_v6(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None, "lo": None, "hi": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v6 requires a seed in config.")
    x_num_train = x_num["train"]
    lo = np.percentile(x_num_train, float(config.get("clip_low_pct", 1.0)), axis=0)
    hi = np.percentile(x_num_train, float(config.get("clip_high_pct", 99.0)), axis=0)
    same = hi <= lo
    if np.any(same):
        lo = np.where(same, x_num_train.min(axis=0), lo)
        hi = np.where(same, x_num_train.max(axis=0), hi)
    clipped = {
        part: np.clip(values, lo, hi).astype(values.dtype)
        for part, values in x_num.items()
    }
    transformer = fit_quantile_normal_transformer(clipped["train"], int(seed))
    transformed = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in clipped.items()
    }
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {"transformer": transformer, "keep_mask": keep_mask, "lo": lo, "hi": hi}
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
