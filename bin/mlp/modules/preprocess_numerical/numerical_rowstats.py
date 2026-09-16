from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def _row_stats(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    missing_fraction = 1.0 - finite.mean(axis=1)
    counts = finite.sum(axis=1).astype(np.float32)
    safe = np.where(finite, values, 0.0).astype(np.float32)
    row_sum = safe.sum(axis=1)
    row_mean = np.divide(row_sum, counts, out=np.zeros_like(row_sum), where=counts > 0)
    centered = np.where(finite, values - row_mean[:, None], 0.0).astype(np.float32)
    row_var = np.divide(
        (centered * centered).sum(axis=1),
        counts,
        out=np.zeros_like(row_sum),
        where=counts > 0,
    )
    row_std = np.sqrt(row_var)
    return np.column_stack(
        [
            np.nan_to_num(missing_fraction, nan=0.0),
            np.nan_to_num(row_mean, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(row_std, nan=0.0, posinf=0.0, neginf=0.0),
        ]
    ).astype(np.float32)


def numerical_preprocess_v7(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Train-only rank-Gauss numerical features plus row-level health descriptors.

    Per-column quantile normalization erases useful row-composition signal
    (e.g. row missingness, raw row-mean, raw row-std). This keeps the standard
    train-only rank-Gauss transform and appends three train-standardized row
    descriptors (missing fraction, raw finite mean, raw finite std) as extra
    columns — helpful when datasets carry meaningful per-row structure.
    """
    if x_num is None:
        return None, {"transformer": None, "keep_mask": None, "stats_center": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v7 requires a seed in config.")
    x_num_train = x_num["train"]
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=int(seed),
    )
    noisy_train = x_num_train + np.random.RandomState(int(seed)).normal(
        0.0, 1e-5, x_num_train.shape
    ).astype(x_num_train.dtype)
    transformer.fit(noisy_train)
    transformed = {
        part: np.nan_to_num(transformer.transform(values).astype(np.float32)).astype(np.float32)
        for part, values in x_num.items()
    }
    base_keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, base_keep_mask] for part, values in transformed.items()}
    stats = {part: _row_stats(values) for part, values in x_num.items()}
    stats_center = stats["train"].mean(axis=0, keepdims=True)
    stats_scale = stats["train"].std(axis=0, keepdims=True)
    stats_scale = np.where(stats_scale == 0.0, 1.0, stats_scale)
    stats = {
        part: ((values - stats_center) / stats_scale).astype(np.float32)
        for part, values in stats.items()
    }
    transformed = {
        part: np.column_stack([values, stats[part]]).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    artifacts = {
        "transformer": transformer,
        "keep_mask": keep_mask,
        "base_keep_mask": base_keep_mask,
        "stats_center": stats_center,
        "stats_scale": stats_scale,
    }
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
