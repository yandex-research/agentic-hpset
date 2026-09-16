from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import (
    filter_constant_columns,
    fit_quantile_normal_transformer,
)


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Quantile-normalized numeric features plus train-fitted missing indicators."""
    if x_num is None:
        return None, {
            "transformer": None,
            "keep_mask": None,
            "missing_indicator_mask": None,
        }
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v4 requires a seed in config.")
    transformer = fit_quantile_normal_transformer(x_num["train"], int(seed))
    transformed = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in x_num.items()
    }
    transformed, keep_mask = filter_constant_columns(transformed)
    missing_indicator_mask = np.isnan(x_num["train"]).any(axis=0)
    if missing_indicator_mask.any():
        for part, values in x_num.items():
            indicators = np.isnan(values[:, missing_indicator_mask]).astype(np.float32)
            transformed[part] = (
                indicators
                if transformed[part].shape[1] == 0
                else np.column_stack([transformed[part], indicators]).astype(np.float32)
            )
    artifacts = {
        "transformer": transformer,
        "keep_mask": keep_mask,
        "missing_indicator_mask": missing_indicator_mask,
    }
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
