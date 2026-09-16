from __future__ import annotations

import numpy as np


def augment_feature_noise(
    x: np.ndarray, *, seed: int, part: str, member_index: int
) -> np.ndarray:
    if part == "train":
        return x
    rng = np.random.default_rng([seed, member_index, len(part), 31])
    scale = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    noise = rng.normal(0.0, 0.05, size=x.shape).astype(np.float32) * scale
    return (x + noise).astype(np.float32)


__all__ = ["augment_feature_noise"]
