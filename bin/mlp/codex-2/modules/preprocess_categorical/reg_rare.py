# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_rare``."""

from __future__ import annotations
from typing import Any
from collections import Counter
import numpy as np

MISSING_KEY = ('__missing__',)


def _key(value: object) -> object:
    if value is None:
        return MISSING_KEY
    try:
        if bool(np.isnan(value)):
            return MISSING_KEY
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fit_maps(x_train: np.ndarray, *, mode: str) -> list[dict[str, object]]:
    maps: list[dict[str, object]] = []
    n_rows, n_cols = x_train.shape
    rare_threshold = max(2, int(np.ceil(n_rows * 0.001)))
    for column in range(n_cols):
        keys = [_key(value) for value in x_train[:, column]]
        counts = Counter(keys)
        if mode == 'frequency':
            ordered = sorted(counts, key=lambda item: (-counts[item], repr(item)))
            mapping = {key: idx for idx, key in enumerate(ordered)}
            unknown = len(mapping)
            missing = mapping.get(MISSING_KEY, unknown)
        elif mode == 'rare':
            frequent = [
                key
                for key, count in counts.items()
                if count >= rare_threshold and key != MISSING_KEY
            ]
            ordered = sorted(frequent, key=lambda item: (-counts[item], repr(item)))
            mapping = {key: idx + 1 for idx, key in enumerate(ordered)}
            unknown = 0
            missing = 0
        elif mode == 'rare_missing':
            frequent = [
                key
                for key, count in counts.items()
                if count >= rare_threshold and key != MISSING_KEY
            ]
            ordered = sorted(frequent, key=lambda item: (-counts[item], repr(item)))
            mapping = {key: idx + 2 for idx, key in enumerate(ordered)}
            unknown = 1
            missing = 0
        else:
            raise ValueError(f'Unknown categorical mode: {mode}')
        maps.append(
            {
                'mapping': mapping,
                'unknown': unknown,
                'missing': missing,
                'rare_threshold': rare_threshold,
            }
        )
    return maps


def _transform(values: np.ndarray, maps: list[dict[str, object]]) -> np.ndarray:
    encoded = np.empty(values.shape, dtype=np.int64)
    for column, spec in enumerate(maps):
        mapping = spec['mapping']
        if not isinstance(mapping, dict):
            raise TypeError('Invalid categorical mapping artifact.')
        unknown = int(spec['unknown'])
        missing = int(spec['missing'])
        for row, value in enumerate(values[:, column]):
            key = _key(value)
            encoded[row, column] = (
                missing if key == MISSING_KEY else int(mapping.get(key, unknown))
            )
    return encoded


def _categorical_preprocess_variant(
    x_cat: dict[str, np.ndarray] | None, *, mode: str
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'variant': mode, 'maps': None})
    if x_cat['train'].shape[1] == 0:
        return (x_cat, {'variant': mode, 'maps': []})
    maps = _fit_maps(x_cat['train'], mode=mode)
    encoded = {part: _transform(values, maps) for part, values in x_cat.items()}
    return (encoded, {'variant': mode, 'maps': maps})


def categorical_preprocess_rare(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    return _categorical_preprocess_variant(x_cat, mode='rare')
