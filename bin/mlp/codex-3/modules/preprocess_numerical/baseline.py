# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``numerical_preprocess_v0``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v0(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None})
    config = config or {}
    seed = int(config.get('seed', 0))
    train = np.asarray(x_num['train'])
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=None,
        random_state=seed,
    )
    noisy_train = train + np.random.RandomState(seed).normal(
        0.0, 1e-05, train.shape
    ).astype(train.dtype)
    transformer.fit(noisy_train)
    transformed = {
        part: np.nan_to_num(transformer.transform(values)).astype(np.float32)
        for part, values in x_num.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'transformer': transformer, 'keep_mask': keep_mask})
    return (transformed, {'transformer': transformer, 'keep_mask': keep_mask})
