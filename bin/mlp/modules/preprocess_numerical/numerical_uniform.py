from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v6(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Train-only uniform rank scaling to [-1, 1].

    Fits a `QuantileTransformer(output_distribution="uniform")` on train only,
    then rescales the uniform output from [0, 1] to [-1, 1]. Compared to the
    normal-output variant, this gives PLR roughly equal-density intervals
    without the extreme tails that a normal rank transform introduces.
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v6 requires a seed in config.")
    x_num_train = x_num["train"]
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
        output_distribution="uniform",
        subsample=1_000_000_000,
        random_state=int(seed),
    )
    noise = np.random.RandomState(int(seed)).normal(0.0, 1e-5, x_num_train.shape).astype(
        x_num_train.dtype
    )
    transformer.fit(x_num_train + noise)
    transformed = {
        part: np.nan_to_num(transformer.transform(values) * 2.0 - 1.0).astype(np.float32)
        for part, values in x_num.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed["train"].shape[1] == 0:
        return None, {"transformer": transformer, "keep_mask": keep_mask}
    return transformed, {"transformer": transformer, "keep_mask": keep_mask}
