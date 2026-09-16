from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def _target_orderings(column: np.ndarray, y: np.ndarray, n_classes: int) -> dict[Any, int]:
    values = np.unique(column)
    fallback = 0.5 if n_classes == 2 else float(y.mean())
    scores = np.array(
        [float(y[column == v].mean()) if (column == v).any() else fallback for v in values]
    )
    order = np.argsort(scores, kind="stable")
    return {values[idx]: int(new_id) for new_id, idx in enumerate(order)}


def categorical_preprocess_v3(
    x_cat: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Target-aware ordinal encoding.

    Each column's categories are sorted by their training-set mean of the
    target (binclass: P(y=1|c); multiclass/regression: mean target value).
    The integer code is then monotone in a target signal — a strictly better
    starting point for downstream embeddings than alphabetical order. Unknown
    val/test categories fall back to a per-column "unknown" bucket.

    Requires `config["y_train"]` (raw training targets). If absent, falls back
    to standard `OrdinalEncoder` behavior.
    """
    if x_cat is None:
        return None, {"encoder": None, "unknown_value": None}
    config = config or {}
    y_train = config.get("y_train")
    n_classes = int(config.get("n_classes", 2))
    if y_train is None:
        unknown_value = np.iinfo(np.int64).max - 3
        encoder = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=unknown_value,
            dtype=np.int64,
        ).fit(x_cat["train"])
        encoded = {
            part: encoder.transform(values).astype(np.int64) for part, values in x_cat.items()
        }
        train_max = (
            encoded["train"].max(axis=0)
            if encoded["train"].shape[1]
            else np.array([], dtype=np.int64)
        )
        for part in ("val", "test"):
            if part not in encoded:
                continue
            for column in range(encoded[part].shape[1]):
                mask = encoded[part][:, column] == unknown_value
                if mask.any():
                    encoded[part][mask, column] = train_max[column] + 1
        return encoded, {"encoder": encoder, "unknown_value": unknown_value}
    train = x_cat["train"]
    n_features = train.shape[1]
    mappings: list[dict[Any, int]] = [
        _target_orderings(train[:, j], y_train, n_classes) for j in range(n_features)
    ]
    fallback = [len(m) for m in mappings]

    def _apply(arr: np.ndarray) -> np.ndarray:
        out = np.empty(arr.shape, dtype=np.int64)
        for j in range(n_features):
            mapping = mappings[j]
            unknown_id = fallback[j]
            col_out = np.empty(arr.shape[0], dtype=np.int64)
            for i, v in enumerate(arr[:, j]):
                col_out[i] = mapping.get(v, unknown_id)
            out[:, j] = col_out
        return out

    encoded = {part: _apply(values) for part, values in x_cat.items()}
    return encoded, {"mappings": mappings, "fallback": fallback}
