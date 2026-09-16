from __future__ import annotations

import numpy as np


def augment_identity(
    x: np.ndarray, *, seed: int, part: str, member_index: int
) -> np.ndarray:
    """No-op augmentation: the member sees the features unchanged."""
    return x


__all__ = ["augment_identity"]
