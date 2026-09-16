from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v1(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Standard z-score normalization for numerical features.

    Uses StandardScaler (subtract mean, divide by std) instead of
    QuantileTransformer. Preserves the original distributional shape
    and is simpler/faster. Works well when features are roughly Gaussian
    or when the model should see the true distribution (e.g., skew, kurtosis).
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    x_num_train = x_num["train"]
    transformer = sklearn.preprocessing.StandardScaler()
    transformer.fit(x_num_train)
    transformed = {part: transformer.transform(values) for part, values in x_num.items()}
    transformed = {
        part: np.nan_to_num(values).astype(np.float32) for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed["train"].shape[1] == 0:
        return None, {"transformer": transformer, "keep_mask": keep_mask}
    return transformed, {"transformer": transformer, "keep_mask": keep_mask}
