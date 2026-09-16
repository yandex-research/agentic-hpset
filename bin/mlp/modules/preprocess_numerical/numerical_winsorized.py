from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Winsorized scaling: clip outliers at [1st, 99th] percentile then standardize.

    First clips each feature to the 1st and 99th percentile of the training
    distribution, then applies standard z-score normalization. This is more
    aggressive than QuantileTransformer (which maps everything smoothly) but
    preserves the bulk distribution shape. Effective when extreme outliers
    distort standardization but you don't want the full nonlinear warping of
    quantile transforms.
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None, "clip_low": None, "clip_high": None}
    x_num_train = x_num["train"]
    # Compute winsorization bounds from training data
    clip_low = np.percentile(x_num_train, 1, axis=0)
    clip_high = np.percentile(x_num_train, 99, axis=0)
    # Clip all splits
    clipped = {
        part: np.clip(values, clip_low, clip_high)
        for part, values in x_num.items()
    }
    # Then standardize
    transformer = sklearn.preprocessing.StandardScaler()
    transformer.fit(clipped["train"])
    transformed = {part: transformer.transform(values) for part, values in clipped.items()}
    transformed = {
        part: np.nan_to_num(values).astype(np.float32) for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed["train"].shape[1] == 0:
        return None, {"transformer": transformer, "keep_mask": keep_mask, "clip_low": clip_low, "clip_high": clip_high}
    return transformed, {"transformer": transformer, "keep_mask": keep_mask, "clip_low": clip_low, "clip_high": clip_high}
