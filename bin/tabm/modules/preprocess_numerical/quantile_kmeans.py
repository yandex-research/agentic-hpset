from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.cluster

from bin.tabm.modules._shared import (
    filter_constant_columns,
    fit_quantile_normal_transformer,
)


def numerical_preprocess_v8(
    x_num: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return None, {"transformer": None, "kmeans": None, "keep_mask": None}
    config = config or {}
    seed = config.get("seed")
    if seed is None:
        raise ValueError("numerical_preprocess_v8 requires a seed in config.")
    transformer = fit_quantile_normal_transformer(x_num["train"], int(seed))
    base = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in x_num.items()
    }
    train_size = base["train"].shape[0]
    n_clusters = min(int(config.get("n_clusters", 16)), max(2, train_size))
    if base["train"].shape[1] == 0 or n_clusters < 2:
        base, keep_mask = filter_constant_columns(base)
        artifacts = {"transformer": transformer, "kmeans": None, "keep_mask": keep_mask}
        if base["train"].shape[1] == 0:
            return None, artifacts
        return base, artifacts
    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(seed),
        n_init=3,
        batch_size=min(1024, train_size),
    )
    kmeans.fit(base["train"])
    augmented: dict[str, np.ndarray] = {}
    for part, values in base.items():
        labels = kmeans.predict(values)
        one_hot = np.zeros((labels.shape[0], n_clusters), dtype=np.float32)
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        augmented[part] = np.concatenate([values.astype(np.float32), one_hot], axis=1)
    augmented, keep_mask = filter_constant_columns(augmented)
    artifacts = {"transformer": transformer, "kmeans": kmeans, "keep_mask": keep_mask}
    if augmented["train"].shape[1] == 0:
        return None, artifacts
    return augmented, artifacts
