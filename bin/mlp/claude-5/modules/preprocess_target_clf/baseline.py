# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``target_preprocess_v0``."""

from __future__ import annotations
from typing import Any
import numpy as np


def target_preprocess_v0(
    y: dict[str, np.ndarray], _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    prepared = {
        part: np.asarray(values).reshape(-1).astype(np.int64)
        for part, values in y.items()
    }
    return (prepared, {'inverse_transform': lambda values: np.asarray(values)})
