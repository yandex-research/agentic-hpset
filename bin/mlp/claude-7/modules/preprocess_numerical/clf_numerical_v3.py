# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v3``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def numerical_preprocess_v3(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """QuantileTransformer + random Gaussian projection augmentation.

    On top of the standard normal-quantile transform, we concatenate a random
    Gaussian projection (deterministic from seed) of the transformed features.
    This injects pre-computed linear feature interactions, giving the
    downstream MLP an inductive bias toward exploring linear combinations of
    raw features in addition to the per-feature embeddings.
    """
    if x_num is None:
        return (None, {'transformer': None, 'keep_mask': None, 'projection': None})
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError('numerical_preprocess_v3 requires a seed in config.')
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
    n_features = transformed['train'].shape[1]
    n_projections = max(min(n_features, 16), 4) if n_features > 0 else 0
    rng = np.random.RandomState(seed + 13)
    projection = (
        rng.normal(
            0.0, 1.0 / np.sqrt(max(n_features, 1)), (n_features, n_projections)
        ).astype(np.float32)
        if n_features > 0 and n_projections > 0
        else None
    )
    if projection is not None:
        augmented = {
            part: np.concatenate([values, values @ projection], axis=1).astype(
                np.float32
            )
            for part, values in transformed.items()
        }
    else:
        augmented = transformed
    keep_mask = np.array(
        [len(np.unique(column)) > 1 for column in augmented['train'].T], dtype=bool
    )
    augmented = {part: values[:, keep_mask] for part, values in augmented.items()}
    if augmented['train'].shape[1] == 0:
        return (
            None,
            {
                'transformer': transformer,
                'keep_mask': keep_mask,
                'projection': projection,
            },
        )
    return (
        augmented,
        {'transformer': transformer, 'keep_mask': keep_mask, 'projection': projection},
    )
