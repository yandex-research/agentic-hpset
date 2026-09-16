from __future__ import annotations

import numpy as np

from .feature_noise import augment_feature_noise
from .identity import augment_identity

# index -> augmentation fn (x, *, seed, part, member_index) -> np.ndarray
AUGMENTATION_MAP = {0: augment_identity, 1: augment_feature_noise}
# basket index -> tuple of augmentation indices mixed across ensemble members
# (0 = identity only / no augmentation; 1 = identity + feature_noise mixed)
AUGMENTATION_BASKETS = {0: (0,), 1: (0, 1)}
AUGMENTATION_NAMES = {0: "identity", 1: "feature_noise"}


def apply_augmentation(
    x: np.ndarray,
    augmentation_idx: int,
    *,
    seed: int,
    part: str,
    member_index: int,
) -> np.ndarray:
    return AUGMENTATION_MAP[augmentation_idx](
        x, seed=seed, part=part, member_index=member_index
    )


__all__ = [
    "AUGMENTATION_MAP",
    "AUGMENTATION_BASKETS",
    "AUGMENTATION_NAMES",
    "apply_augmentation",
    "augment_identity",
    "augment_feature_noise",
]
