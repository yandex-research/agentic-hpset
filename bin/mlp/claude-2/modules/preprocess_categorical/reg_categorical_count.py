# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing

_RARE_FRACTION = 0.005


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None, 'rare_threshold': None})
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype=np.int64
    ).fit(x_cat['train'])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    train_max = (
        encoded['train'].max(axis=0)
        if encoded['train'].shape[1]
        else np.array([], dtype=np.int64)
    )
    for part in ('val', 'test'):
        for column in range(encoded[part].shape[1]):
            mask = encoded[part][:, column] == unknown_value
            if mask.any():
                encoded[part][mask, column] = train_max[column] + 1
    n_features = encoded['train'].shape[1]
    train_size = encoded['train'].shape[0]
    if n_features == 0 or train_size == 0:
        return (
            encoded,
            {'encoder': encoder, 'unknown_value': unknown_value, 'rare_threshold': 0},
        )
    rare_threshold = max(1, int(round(_RARE_FRACTION * train_size)))
    remap_lookups: list[dict[int, int]] = []
    for column in range(n_features):
        col_train = encoded['train'][:, column]
        max_id = int(col_train.max(initial=0)) + 1
        for part in ('val', 'test'):
            if encoded[part].shape[0]:
                max_id = max(max_id, int(encoded[part][:, column].max(initial=0)) + 1)
        bincount = np.bincount(col_train, minlength=max_id)
        kept_ids = np.where(bincount >= rare_threshold)[0]
        if kept_ids.size == 0:
            kept_ids = np.array([int(np.argmax(bincount))], dtype=np.int64)
        rare_id = int(kept_ids.size)
        remap = {int(orig): rank for rank, orig in enumerate(kept_ids.tolist())}
        remap_lookups.append({'map': remap, 'rare_id': rare_id, 'size': rare_id + 1})
    transformed: dict[str, np.ndarray] = {}
    for part in ('train', 'val', 'test'):
        out = np.zeros_like(encoded[part])
        for column in range(n_features):
            entry = remap_lookups[column]
            mapping = entry['map']
            rare_id = entry['rare_id']
            col = encoded[part][:, column]
            remapped = np.fromiter(
                (mapping.get(int(v), rare_id) for v in col.tolist()),
                dtype=np.int64,
                count=col.shape[0],
            )
            out[:, column] = remapped
        transformed[part] = out
    return (
        transformed,
        {
            'encoder': encoder,
            'unknown_value': unknown_value,
            'rare_threshold': rare_threshold,
            'remap_lookups': remap_lookups,
        },
    )
