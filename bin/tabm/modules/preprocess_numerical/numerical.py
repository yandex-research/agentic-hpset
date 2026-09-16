from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import (
    filter_constant_columns,
    fit_quantile_normal_transformer,
)


def numerical_preprocess_v0(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v0 requires a seed in config.")
    transformer = fit_quantile_normal_transformer(x_num["train"], int(seed))
    transformed = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in x_num.items()
    }
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {"transformer": transformer, "keep_mask": keep_mask}
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
