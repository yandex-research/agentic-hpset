from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import sklearn.preprocessing


def _stable_hash(value: object, salt: int) -> int:
    digest = hashlib.blake2b(
        repr(value).encode("utf-8"),
        digest_size=8,
        key=salt.to_bytes(8, "little"),
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def categorical_preprocess_v2(
    x_cat: dict[str, np.ndarray] | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return None, {
            "buckets_per_feature": None,
            "salts": None,
            "use_hash": None,
            "fallback_encoders": None,
            "bucket_maps": None,
            "unknown_indices": None,
        }
    config = config or {}
    seed = int(config.get("seed", 0))
    max_buckets = int(config.get("hash_max_buckets", 64))
    threshold = int(config.get("hash_threshold_cardinality", 32))
    n_features = x_cat["train"].shape[1]
    base_unknown = np.iinfo(np.int64).max - 3
    buckets_per_feature: list[int] = []
    salts: list[int] = []
    use_hash: list[bool] = []
    encoders: list[sklearn.preprocessing.OrdinalEncoder | None] = []
    bucket_maps: list[dict[int, int] | None] = []
    unknown_indices: list[int] = []
    for feature_idx in range(n_features):
        cardinality = np.unique(x_cat["train"][:, feature_idx]).shape[0]
        if cardinality > threshold:
            n_buckets = min(max_buckets, max(2, cardinality))
            train_buckets = np.array(
                [
                    _stable_hash(value, seed * 1009 + feature_idx * 31 + 7) % n_buckets
                    for value in x_cat["train"][:, feature_idx]
                ],
                dtype=np.int64,
            )
            observed_buckets = np.sort(np.unique(train_buckets))
            bucket_to_id = {
                int(bucket): idx for idx, bucket in enumerate(observed_buckets)
            }
            buckets_per_feature.append(n_buckets)
            salts.append(seed * 1009 + feature_idx * 31 + 7)
            use_hash.append(True)
            encoders.append(None)
            bucket_maps.append(bucket_to_id)
            unknown_indices.append(len(bucket_to_id))
        else:
            buckets_per_feature.append(cardinality)
            salts.append(0)
            use_hash.append(False)
            encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=base_unknown,
                dtype=np.int64,
            ).fit(x_cat["train"][:, feature_idx : feature_idx + 1])
            encoders.append(encoder)
            bucket_maps.append(None)
            unknown_indices.append(len(encoder.categories_[0]))
    output: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        result = np.zeros((values.shape[0], n_features), dtype=np.int64)
        for feature_idx in range(n_features):
            unknown_index = unknown_indices[feature_idx]
            if use_hash[feature_idx]:
                bucket_to_id = bucket_maps[feature_idx]
                if bucket_to_id is None:
                    raise RuntimeError("Missing hash bucket map.")
                buckets = np.array(
                    [
                        _stable_hash(value, salts[feature_idx])
                        % buckets_per_feature[feature_idx]
                        for value in values[:, feature_idx]
                    ],
                    dtype=np.int64,
                )
                result[:, feature_idx] = np.array(
                    [
                        bucket_to_id.get(int(bucket), unknown_index)
                        for bucket in buckets
                    ],
                    dtype=np.int64,
                )
                continue
            encoder = encoders[feature_idx]
            if encoder is None:
                raise RuntimeError("Missing fallback encoder.")
            encoded = (
                encoder.transform(values[:, feature_idx : feature_idx + 1])
                .astype(np.int64)
                .reshape(-1)
            )
            result[:, feature_idx] = np.where(
                encoded == base_unknown, unknown_index, encoded
            )
        output[part] = result
    return output, {
        "buckets_per_feature": buckets_per_feature,
        "salts": salts,
        "use_hash": use_hash,
        "fallback_encoders": encoders,
        "bucket_maps": bucket_maps,
        "unknown_indices": unknown_indices,
    }
