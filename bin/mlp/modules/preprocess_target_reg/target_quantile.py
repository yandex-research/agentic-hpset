from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def target_preprocess_v1(
    y_raw: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Quantile-based target transform (maps target to normal distribution).

    Uses QuantileTransformer to normalize the target distribution. This helps
    when target distributions are heavily skewed, making the loss landscape
    more symmetric and easier to optimize. The inverse transform is applied
    at evaluation time to recover predictions in the original scale.
    """
    config = config or {}
    seed = config.get("seed", 0)
    n_train = y_raw["train"].shape[0]
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(n_train // 30, 1000), 10),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=seed,
    )
    # QuantileTransformer expects 2D input
    transformer.fit(y_raw["train"].reshape(-1, 1))
    y = {
        part: transformer.transform(values.reshape(-1, 1)).ravel().astype(np.float32)
        for part, values in y_raw.items()
    }
    y = {part: np.nan_to_num(values).astype(np.float32) for part, values in y.items()}

    def inverse_transform(values: np.ndarray) -> np.ndarray:
        return transformer.inverse_transform(values.reshape(-1, 1)).ravel()

    return y, {
        "transformer": transformer,
        "inverse_transform": inverse_transform,
    }
