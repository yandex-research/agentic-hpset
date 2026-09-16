# ruff: noqa
"""Standalone implementation for ``numerical_preprocess_v4``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(parts) == 1:
        return {part: values.astype(np.float32) for part, values in parts[0].items()}
    return {
        part: np.concatenate([block[part] for block in parts], axis=1).astype(
            np.float32
        )
        for part in parts[0]
    }


def _drop_constant(
    transformed: dict[str, np.ndarray], artifacts: dict[str, object]
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    train = np.nan_to_num(transformed['train'], nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = train.max(axis=0) - train.min(axis=0) > 1e-12
    cleaned = {
        part: np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)[
            :, keep_mask
        ]
        for part, values in transformed.items()
    }
    artifacts = {**artifacts, 'keep_mask': keep_mask}
    if cleaned['train'].shape[1] == 0:
        return (None, artifacts)
    return (cleaned, artifacts)


def _median_impute(
    x_num: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train = x_num['train'].astype(np.float32, copy=False)
    median = np.nanmedian(train, axis=0).astype(np.float32)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    return (
        {
            part: np.where(np.isnan(values), median, values).astype(np.float32)
            for part, values in x_num.items()
        },
        median,
    )


def _missing_indicators(
    x_num: dict[str, np.ndarray], train_only: bool = True
) -> dict[str, np.ndarray] | None:
    if train_only:
        mask = np.isnan(x_num['train']).any(axis=0)
    else:
        mask = np.column_stack(
            [np.isnan(values).any(axis=0) for values in x_num.values()]
        ).any(axis=1)
    if not bool(mask.any()):
        return None
    return {
        part: np.isnan(values[:, mask]).astype(np.float32)
        for part, values in x_num.items()
    }


def _n_quantiles(n_rows: int) -> int:
    return min(n_rows, max(min(n_rows // 30, 1000), 10))


def _seed(config: dict[str, Any] | None, name: str) -> int:
    config = config or {}
    seed = config.get('seed')
    if seed is None:
        raise ValueError(f'{name} requires a seed in config.')
    return int(seed)


def _standard_stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-06, std, 1.0).astype(np.float32)
    return (mean, std)


def numerical_preprocess_v4(
    x_num: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Gaussian rank features plus winsorized raw residual and missing flags."""
    if x_num is None:
        return (None, {'variant': 'normal_rank_raw_missing', 'keep_mask': None})
    seed = _seed(config, 'numerical_preprocess_v4')
    imputed, median = _median_impute(x_num)
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=_n_quantiles(imputed['train'].shape[0]),
        output_distribution='normal',
        subsample=1000000000,
        random_state=seed,
    ).fit(imputed['train'])
    ranked = {
        part: transformer.transform(values).astype(np.float32)
        for part, values in imputed.items()
    }
    lo, hi = np.quantile(imputed['train'], [0.005, 0.995], axis=0)
    clipped_train = np.clip(imputed['train'], lo, hi)
    mean, std = _standard_stats(clipped_train)
    raw = {
        part: (np.clip(values, lo, hi) - mean) / std for part, values in imputed.items()
    }
    blocks = [ranked, raw]
    indicators = _missing_indicators(x_num)
    if indicators is not None:
        blocks.append(indicators)
    return _drop_constant(
        _concat(blocks),
        {
            'variant': 'normal_rank_raw_missing',
            'median': median,
            'transformer': transformer,
            'lo': lo,
            'hi': hi,
            'mean': mean,
            'std': std,
        },
    )
