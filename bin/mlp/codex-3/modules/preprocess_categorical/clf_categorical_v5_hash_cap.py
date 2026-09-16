# ruff: noqa
"""Standalone implementation for ``categorical_preprocess_v5``."""

from __future__ import annotations
from typing import Any
import hashlib
import numpy as np


def _hash_value(value: object, cap: int) -> int:
    digest = hashlib.blake2b(str(value).encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % cap


def categorical_preprocess_v5(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'caps': None})
    train = x_cat['train']
    caps = []
    for column in range(train.shape[1]):
        n_unique = len(np.unique(train[:, column]))
        cap = min(512, max(8, int(np.ceil(4.0 * np.sqrt(max(n_unique, 1))))))
        caps.append(cap)
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty(values.shape, dtype=np.int64)
        for column, cap in enumerate(caps):
            out[:, column] = [_hash_value(value, cap) for value in values[:, column]]
        encoded[part] = out
    return (encoded, {'caps': caps})
