from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v2(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Yeo-Johnson power transformation for numerical features.

    Finds the optimal power parameter per feature to make distributions more
    Gaussian-like, then standardizes. Unlike QuantileTransformer, this
    preserves monotonicity and distributional shape information (skew, kurtosis).
    Handles both positive and negative values (unlike Box-Cox).
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    x_num_train = x_num["train"]
    transformer = sklearn.preprocessing.PowerTransformer(
        method="yeo-johnson", standardize=True
    )
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
