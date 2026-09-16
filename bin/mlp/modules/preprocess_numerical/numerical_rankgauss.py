from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special


def _rank_gauss_train(column: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(column)
    order = np.argsort(column, kind="stable")
    ranks_sorted = (np.arange(n, dtype=np.float64) + 1.0) / (n + 1.0)
    ranks_sorted = ranks_sorted * (1.0 - 2.0 * eps) + eps
    gauss_sorted = scipy.special.ndtri(ranks_sorted).astype(np.float32)
    sorted_values = column[order]
    return sorted_values, gauss_sorted


def _apply(column: np.ndarray, sorted_values: np.ndarray, gauss_sorted: np.ndarray) -> np.ndarray:
    n = len(sorted_values)
    if n == 0:
        return np.zeros_like(column, dtype=np.float32)
    idx = np.searchsorted(sorted_values, column, side="left")
    idx = np.clip(idx, 0, n - 1)
    return gauss_sorted[idx].astype(np.float32)


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """RankGauss: replace each numeric feature with its empirical-rank
    mapped through Phi^{-1} (inverse standard-normal CDF).

    Robust to outliers and skewed distributions; output is approximately
    N(0, 1). Val/test values are mapped via searchsorted against the sorted
    training values.
    """
    if x_num is None:
        return None, {"per_feature": None, "keep_mask": None}
    config = config or {}
    seed = int(config.get("seed", 0))
    rng = np.random.RandomState(seed)
    train = x_num["train"].astype(np.float64)
    n_features = train.shape[1]
    eps = 1e-6
    per_feature: list[dict[str, np.ndarray]] = []
    transformed_train = np.empty_like(train, dtype=np.float32)
    for j in range(n_features):
        col = train[:, j] + rng.normal(0.0, 1e-7, train.shape[0])
        sv, gv = _rank_gauss_train(col, eps)
        per_feature.append({"sorted_values": sv, "gauss_sorted": gv})
        transformed_train[:, j] = _apply(train[:, j], sv, gv)
    transformed: dict[str, np.ndarray] = {"train": transformed_train}
    for part, values in x_num.items():
        if part == "train":
            continue
        v_arr = values.astype(np.float64)
        out = np.empty_like(v_arr, dtype=np.float32)
        for j in range(n_features):
            out[:, j] = _apply(v_arr[:, j], per_feature[j]["sorted_values"], per_feature[j]["gauss_sorted"])
        transformed[part] = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed["train"].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed["train"].shape[1] == 0:
        return None, {"per_feature": per_feature, "keep_mask": keep_mask}
    return transformed, {"per_feature": per_feature, "keep_mask": keep_mask}
