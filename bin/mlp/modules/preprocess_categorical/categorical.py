from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.preprocessing


def categorical_preprocess_v0(
    x_cat: dict[str, np.ndarray] | None,
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return None, {"encoder": None, "unknown_value": None}
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype=np.int64,
    ).fit(x_cat["train"])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    train_max = (
        encoded["train"].max(axis=0)
        if encoded["train"].shape[1]
        else np.array([], dtype=np.int64)
    )
    for part in ("val", "test"):
        for column in range(encoded[part].shape[1]):
            mask = encoded[part][:, column] == unknown_value
            if mask.any():
                encoded[part][mask, column] = train_max[column] + 1
    return encoded, {"encoder": encoder, "unknown_value": unknown_value}
