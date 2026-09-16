# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v1``."""

from __future__ import annotations
from typing import Any
import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None, config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None, 'rare_threshold': None})
    config = config or {}
    rare_threshold = int(config.get('rare_threshold', 5))
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
    rare_maps: list[dict[int, int]] = []
    new_cardinalities: list[int] = []
    for col_idx in range(encoded['train'].shape[1]):
        col_train = encoded['train'][:, col_idx]
        max_idx = (
            int(
                max(
                    col_train.max(),
                    encoded['val'][:, col_idx].max(),
                    encoded['test'][:, col_idx].max(),
                )
            )
            + 1
        )
        counts = np.bincount(col_train, minlength=max_idx)
        rare_mask = counts < rare_threshold
        rare_id = int((~rare_mask).sum())
        new_id = -1
        remap: dict[int, int] = {}
        for i, is_rare in enumerate(rare_mask):
            if is_rare:
                remap[i] = rare_id
            else:
                new_id += 1
                remap[i] = new_id
        rare_maps.append(remap)
        new_cardinalities.append(rare_id + 1)

    def remap_split(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for col_idx in range(arr.shape[1]):
            remap = rare_maps[col_idx]
            rare_id = new_cardinalities[col_idx] - 1
            col = out[:, col_idx]
            mapped = np.array([remap.get(int(v), rare_id) for v in col], dtype=np.int64)
            out[:, col_idx] = mapped
        return out

    remapped = {part: remap_split(arr) for part, arr in encoded.items()}
    return (
        remapped,
        {
            'encoder': encoder,
            'unknown_value': unknown_value,
            'rare_threshold': rare_threshold,
            'new_cardinalities': new_cardinalities,
        },
    )
