from __future__ import annotations

from typing import Any

import numpy as np

from bin.tabm.modules._shared import column_centers, fill_nans, filter_constant_columns


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Standard z-score scaling with NaN imputation, no quantile transform.

    Hypothesis: the default quantile transform destroys distance information that
    PLR embeddings could otherwise exploit when the raw distribution is already
    near-Gaussian (e.g., engineered or already-transformed features). Standardize
    preserves geometry while still keeping gradients well-scaled.
    """
    if x_num is None:
        return None, {"mean": None, "std": None, "keep_mask": None}
    train = np.asarray(x_num["train"], dtype=np.float64)
    mean = column_centers(train, np.nanmean)
    train_filled = fill_nans(train, mean)
    std = train_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    transformed: dict[str, np.ndarray] = {
        part: ((fill_nans(np.asarray(values, dtype=np.float64), mean) - mean) / std).astype(np.float32)
        for part, values in x_num.items()
    }
    transformed, keep_mask = filter_constant_columns(transformed)
    artifacts = {"mean": mean, "std": std, "keep_mask": keep_mask}
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
