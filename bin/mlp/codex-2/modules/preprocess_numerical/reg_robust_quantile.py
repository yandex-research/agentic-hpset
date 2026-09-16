# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_robust_quantile``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _as_float32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _drop_constant(
    transformed: dict[str, np.ndarray], *, artifact: dict[str, object]
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    train = transformed['train']
    keep_mask = np.array([np.nanstd(column) > 1e-07 for column in train.T], dtype=bool)
    artifact['keep_mask'] = keep_mask
    if keep_mask.size == 0 or not keep_mask.any():
        return (None, artifact)
    return (
        {
            part: values[:, keep_mask].astype(np.float32)
            for part, values in transformed.items()
        },
        artifact,
    )


def _empty_artifacts(name: str) -> dict[str, object]:
    return {'variant': name, 'keep_mask': None}


def _impute_with_medians(
    x_num: dict[str, np.ndarray], medians: np.ndarray
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        values = _as_float32(values).copy()
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(medians, np.where(missing)[1])
        result[part] = values
    return result


def _quantile_transformer(
    x_train: np.ndarray, *, seed: int, output_distribution: str, divisor: int = 20
) -> sklearn.preprocessing.QuantileTransformer:
    n_train = x_train.shape[0]
    n_quantiles = min(max(n_train // divisor, 10), 2048, n_train)
    n_quantiles = max(n_quantiles, 1)
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=1000000000,
        random_state=seed,
    )


def _train_medians(x_train: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(x_train, axis=0).astype(np.float32)
    return np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)


def numerical_preprocess_robust_quantile(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_num is None:
        return (None, _empty_artifacts('robust_quantile'))
    config = config or {}
    seed = int(config.get('seed', 0))
    medians = _train_medians(_as_float32(x_num['train']))
    imputed = _impute_with_medians(x_num, medians)
    transformer = _quantile_transformer(
        imputed['train'], seed=seed, output_distribution='normal', divisor=12
    )
    transformer.fit(imputed['train'])
    transformed = {
        part: np.clip(transformer.transform(values), -5.0, 5.0).astype(np.float32)
        for part, values in imputed.items()
    }
    return _drop_constant(
        transformed,
        artifact={
            'variant': 'robust_quantile',
            'medians': medians,
            'transformer': transformer,
        },
    )
