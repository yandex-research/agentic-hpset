from __future__ import annotations

from typing import Any

import numpy as np


def _hash_encode(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Hash integer category values into fixed-size buckets.

    Uses a simple multiplicative hash (Knuth's method) that's fast and
    produces reasonably uniform bucket assignments.
    """
    GOLDEN = np.uint64(0x9E3779B97F4A7C15)
    hashed = (values.astype(np.uint64) * GOLDEN).astype(np.int64) % n_buckets
    return hashed.astype(np.int64)


def categorical_preprocess_v1(
    x_cat: dict[str, np.ndarray] | None,
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Hash encoding for categorical features.

    Maps categories to a fixed number of hash buckets per feature. This:
    - Naturally handles unseen categories (they just hash to some bucket)
    - Reduces cardinality for high-cardinality features
    - Is collision-robust (similar categories may share a bucket, providing
      implicit regularization)

    Common in production ML systems (Vowpal Wabbit, ad-click models).
    Uses 64 buckets per feature as a reasonable default.
    """
    if x_cat is None:
        return None, {"encoder": None, "n_buckets": None}
    n_buckets = 64
    # First do ordinal encoding to get consistent integer values
    from sklearn.preprocessing import OrdinalEncoder
    unknown_value = np.iinfo(np.int64).max - 3
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype=np.int64,
    ).fit(x_cat["train"])
    encoded = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in x_cat.items()
    }
    # Apply hash encoding to each column
    hashed = {
        part: np.column_stack([
            _hash_encode(encoded[part][:, col], n_buckets)
            for col in range(encoded[part].shape[1])
        ]) if encoded[part].shape[1] > 0 else encoded[part]
        for part in encoded
    }
    cardinalities = [n_buckets] * int(encoded["train"].shape[1])
    return hashed, {
        "encoder": encoder,
        "n_buckets": n_buckets,
        "cardinalities": cardinalities,
    }
