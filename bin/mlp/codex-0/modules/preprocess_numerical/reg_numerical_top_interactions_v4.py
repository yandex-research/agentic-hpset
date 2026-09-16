# ruff: noqa
"""Standalone implementation for ``numerical_top_interactions_v4``."""

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
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v0 requires a seed in config.')
    x_num_train = x_num['train']
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    )
    noisy_train = x_num_train + np.random.RandomState(seed).normal(
        0.0, 1e-05, x_num_train.shape
    ).astype(x_num_train.dtype)
    transformer.fit(noisy_train)
    transformed = {
        part: transformer.transform(values) for part, values in x_num.items()
    }
    transformed = {
        part: np.nan_to_num(values).astype(np.float32)
        for part, values in transformed.items()
    }
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in transformed['train'].T], dtype=bool
    )
    transformed = {part: values[:, keep_mask] for part, values in transformed.items()}
    if transformed['train'].shape[1] == 0:
        return (None, {'transformer': transformer, 'keep_mask': keep_mask})
    return (transformed, {'transformer': transformer, 'keep_mask': keep_mask})


def numerical_top_interactions_v4(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, {'base': None, 'pairs': []})
    base, artifacts = numerical_preprocess_v0(x_num, config)
    if base is None or base['train'].shape[1] < 2:
        return (base, {'base': artifacts, 'pairs': []})
    train = base['train']
    corr = np.nan_to_num(np.corrcoef(train, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    rows, cols = np.triu_indices(train.shape[1], k=1)
    scores = np.abs(corr[rows, cols])
    n_pairs = min(32, len(scores))
    if n_pairs == 0:
        return (base, {'base': artifacts, 'pairs': []})
    selected = np.argpartition(scores, -n_pairs)[-n_pairs:]
    pairs = [
        (int(rows[i]), int(cols[i]))
        for i in selected[np.argsort(scores[selected])[::-1]]
    ]
    interactions = {
        part: np.column_stack([values[:, i] * values[:, j] for i, j in pairs]).astype(
            np.float32
        )
        for part, values in base.items()
    }
    return (
        {
            part: np.column_stack([base[part], interactions[part]]).astype(np.float32)
            for part in base
        },
        {'base': artifacts, 'pairs': pairs},
    )
