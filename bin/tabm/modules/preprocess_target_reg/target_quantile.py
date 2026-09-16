from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from bin.tabm.modules._shared import fit_quantile_normal_transformer


def _inverse_transform(values: np.ndarray, transformer: QuantileTransformer) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    inverted = transformer.inverse_transform(arr).reshape(-1)
    return inverted.astype(np.float32)


def target_preprocess_v2(
    y_raw: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Map y to standard normal via empirical quantiles, then learn in that space.

    Hypothesis: Gaussianizing the target removes heavy tails and skew that
    inflate MSE gradients on extreme labels, letting the model spend capacity
    on the bulk of the distribution. Inverse-transforming predictions before
    metric computation keeps the original loss landscape intact.
    """
    config = config or {}
    seed = config.get("seed", 0)
    train = y_raw["train"].reshape(-1, 1).astype(np.float64)
    transformer = fit_quantile_normal_transformer(train, int(seed))
    y = {
        part: transformer.transform(values.reshape(-1, 1).astype(np.float64))
        .reshape(-1)
        .astype(np.float32)
        for part, values in y_raw.items()
    }
    # Replace any infinities introduced by the tail of the normal CDF.
    y = {part: np.nan_to_num(values, nan=0.0, posinf=5.0, neginf=-5.0) for part, values in y.items()}
    return y, {
        "transformer": transformer,
        "inverse_transform": lambda values, transformer=transformer: _inverse_transform(
            values, transformer
        ),
    }
