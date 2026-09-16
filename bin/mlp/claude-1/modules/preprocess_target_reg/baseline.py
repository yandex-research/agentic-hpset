# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``target_preprocess_v0``."""

from __future__ import annotations
from typing import Any
import numpy as np


def target_preprocess_v0(
    y: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = np.asarray(y['train'], dtype=np.float64).reshape(-1)
    mean = float(train.mean())
    std = float(train.std())
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    transformed = {
        part: ((np.asarray(values).reshape(-1) - mean) / std).astype(np.float32)
        for part, values in y.items()
    }

    def inverse_transform(values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * std + mean

    return (
        transformed,
        {'mean': mean, 'std': std, 'inverse_transform': inverse_transform},
    )
