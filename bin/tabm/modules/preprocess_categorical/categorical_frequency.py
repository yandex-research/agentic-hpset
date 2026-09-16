from __future__ import annotations

from typing import Any

import numpy as np


def _build_column_mapping(column: np.ndarray, min_count: int) -> dict[object, int]:
    """Map each value to a contiguous int id sorted by descending frequency.

    Values whose count is below ``min_count`` are merged into the same bucket
    (``rare_id``), reducing one-hot dimensionality. Unseen values at val/test
    time are mapped to a dedicated unknown id (returned in the artifacts).
    """
    values, counts = np.unique(column, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    values = values[order]
    counts = counts[order]
    rare_id = 0
    mapping: dict[object, int] = {}
    next_id = 1
    has_rare = False
    for value, count in zip(values, counts):
        if count < min_count:
            mapping[value] = rare_id
            has_rare = True
        else:
            mapping[value] = next_id
            next_id += 1
    if not has_rare:
        # Re-pack: there is no rare bucket, so shift ids down to start at 0.
        mapping = {value: idx - 1 for value, idx in mapping.items()}
        next_id -= 1
        rare_id = next_id  # Reserved for unknowns at inference.
        next_id += 1
    return mapping, rare_id, next_id


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Frequency-rank encode categories; merge rare levels into a shared bucket.

    Hypothesis: high-cardinality categorical features explode TabM's input width
    via one-hot, while many of those levels have only a handful of training rows
    and look like noise to the optimizer. Bucketing rare levels (count < threshold)
    into one ``rare`` token shrinks the input dimension, regularizes via shared
    representation, and removes the long tail. Unknown levels at inference reuse
    the ``rare`` slot, producing a graceful fallback.
    """
    if x_cat is None:
        return None, {"mappings": None, "min_count": None}
    config = config or {}
    n_train = int(x_cat["train"].shape[0])
    min_count = max(2, int(round(n_train * 0.001)))

    n_columns = x_cat["train"].shape[1]
    mappings: list[dict[object, int]] = []
    rare_ids: list[int] = []
    cardinalities: list[int] = []
    for col in range(n_columns):
        mapping, rare_id, cardinality = _build_column_mapping(
            x_cat["train"][:, col], min_count
        )
        mappings.append(mapping)
        rare_ids.append(rare_id)
        cardinalities.append(cardinality)

    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        out = np.empty_like(values, dtype=np.int64)
        for col in range(n_columns):
            mapping = mappings[col]
            rare_id = rare_ids[col]
            column = values[:, col]
            out[:, col] = np.array(
                [mapping.get(v, rare_id) for v in column], dtype=np.int64
            )
        encoded[part] = out
    return encoded, {
        "mappings": mappings,
        "rare_ids": rare_ids,
        "cardinalities": cardinalities,
        "min_count": min_count,
    }
