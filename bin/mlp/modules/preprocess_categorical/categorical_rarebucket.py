from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def _build_rare_lookup(column: np.ndarray, min_count: int) -> dict[Any, int]:
    values, counts = np.unique(column, return_counts=True)
    keep = counts >= min_count
    next_id = 0
    mapping: dict[Any, int] = {}
    for v, k in zip(values.tolist(), keep.tolist()):
        if k:
            mapping[v] = next_id
            next_id += 1
    rare_id = next_id
    for v, k in zip(values.tolist(), keep.tolist()):
        if not k:
            mapping[v] = rare_id
    return mapping


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Ordinal encoding with per-column rare-category collapsing.

    Training-time categories with count < `min_count` (default 5) collapse
    into a shared RARE bucket per column. Val/test unknown values also route
    to that bucket. Reduces effective cardinality and noise from values seen
    only a few times.
    """
    if x_cat is None:
        return None, {"encoder": None, "unknown_value": None}
    config = config or {}
    min_count = int(config.get("rare_min_count", 5))
    train = x_cat["train"]
    n_features = train.shape[1]
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype=np.int64,
    ).fit(train)
    encoded = {
        part: encoder.transform(values).astype(np.int64) for part, values in x_cat.items()
    }
    encoded_remap: dict[str, np.ndarray] = {
        part: np.empty_like(arr) for part, arr in encoded.items()
    }
    rare_buckets: list[int] = []
    for j in range(n_features):
        col_train = encoded["train"][:, j]
        mapping = _build_rare_lookup(col_train, min_count)
        rare_id = max(mapping.values()) if mapping else 0
        rare_buckets.append(rare_id)
        max_train = int(col_train.max()) if col_train.size else 0
        lookup = np.full(max_train + 2, rare_id, dtype=np.int64)
        for v, eid in mapping.items():
            if 0 <= int(v) <= max_train:
                lookup[int(v)] = eid
        for part, arr in encoded.items():
            col = arr[:, j]
            unknown_mask = col == unknown_value
            col_safe = np.where(unknown_mask, 0, col)
            in_range = col_safe < lookup.size
            out = np.where(in_range, lookup[np.clip(col_safe, 0, lookup.size - 1)], rare_id)
            out = np.where(unknown_mask, rare_id, out)
            encoded_remap[part][:, j] = out
    return encoded_remap, {
        "encoder": encoder,
        "min_count": min_count,
        "rare_buckets": rare_buckets,
    }
