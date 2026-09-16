from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats
import sklearn.preprocessing


def _quantile_transform(x_num: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    train = x_num["train"]
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(train.shape[0] // 30, 1000), 10),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=seed,
    )
    noisy_train = train + np.random.RandomState(seed).normal(0.0, 1e-5, train.shape).astype(train.dtype)
    transformer.fit(noisy_train)
    out = {part: transformer.transform(values) for part, values in x_num.items()}
    return {part: np.nan_to_num(v).astype(np.float32) for part, v in out.items()}


def numerical_preprocess_v5(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Quantile transform + engineered log1p copies of skewed columns.

    Skew is measured on training data; columns with |skew| > 1 contribute an
    extra `sign(x) * log1p(|x|)` feature (standardized using train stats),
    concatenated alongside the quantile-transformed base. Gives downstream
    layers both the smooth and the order-preserving log-compressed views of
    heavy-tailed features.
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v5 requires a seed in config.")
    base = _quantile_transform(x_num, int(seed))
    train = x_num["train"]
    skews = np.array(
        [
            float(scipy.stats.skew(train[:, j], bias=False, nan_policy="omit"))
            if np.unique(train[:, j]).size > 1
            else 0.0
            for j in range(train.shape[1])
        ],
        dtype=np.float64,
    )
    skew_mask = np.abs(skews) > 1.0
    extras: dict[str, np.ndarray] = {}
    log_mean: np.ndarray | None = None
    log_std: np.ndarray | None = None
    for part, values in x_num.items():
        skewed = values[:, skew_mask]
        log_features = np.sign(skewed) * np.log1p(np.abs(skewed))
        if part == "train":
            log_mean = log_features.mean(axis=0, keepdims=True)
            log_std = log_features.std(axis=0, keepdims=True)
            log_std = np.where(log_std == 0, 1.0, log_std)
        log_features = (log_features - log_mean) / log_std
        extras[part] = np.nan_to_num(log_features.astype(np.float32))
    combined = {part: np.concatenate([base[part], extras[part]], axis=1) for part in base}
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in combined["train"].T], dtype=bool
    )
    combined = {part: values[:, keep_mask] for part, values in combined.items()}
    if combined["train"].shape[1] == 0:
        return None, {"keep_mask": keep_mask, "skew_mask": skew_mask}
    return combined, {"keep_mask": keep_mask, "skew_mask": skew_mask}
