from __future__ import annotations

from typing import Any

import numpy as np


def _build_rank_lookup(column: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    noise = np.random.RandomState(seed).normal(0.0, 1e-9, column.shape).astype(column.dtype)
    sorted_values = np.sort(column + noise)
    ranks = (np.arange(sorted_values.shape[0], dtype=np.float64) + 0.5) / sorted_values.shape[0]
    return sorted_values.astype(np.float64), ranks.astype(np.float32)


def _apply_rank(
    column: np.ndarray, sorted_values: np.ndarray, ranks: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(sorted_values, column.astype(np.float64), side="left")
    indices = np.clip(indices, 0, len(ranks) - 1)
    return ranks[indices].astype(np.float32)


def numerical_preprocess_v7(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return None, {"sorted": None, "ranks": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v7 requires a seed in config.")
    x_num_train = x_num["train"]
    sorted_per_feature: list[np.ndarray] = []
    ranks_per_feature: list[np.ndarray] = []
    for feature_idx in range(x_num_train.shape[1]):
        sorted_values, ranks = _build_rank_lookup(x_num_train[:, feature_idx], seed + feature_idx)
        sorted_per_feature.append(sorted_values)
        ranks_per_feature.append(ranks)
    transformed: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        transformed[part] = np.stack(
            [
                _apply_rank(values[:, feature_idx], sorted_per_feature[feature_idx], ranks_per_feature[feature_idx])
                for feature_idx in range(x_num_train.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
        transformed[part] = transformed[part] * 2.0 - 1.0
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    artifacts = {
        "sorted": sorted_per_feature,
        "ranks": ranks_per_feature,
        "keep_mask": keep_mask,
    }
    if transformed["train"].shape[1] == 0:
        return None, artifacts
    return transformed, artifacts
