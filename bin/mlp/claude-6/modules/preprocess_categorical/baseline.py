# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``categorical_preprocess_v0``."""

from __future__ import annotations
from typing import Any
import numpy as np
from ...core import ordinal_encode_categories


def categorical_preprocess_v0(
    x_cat: dict[str, np.ndarray] | None, _config: dict[str, Any] | None = None
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    if x_cat is None:
        return (None, {'encoder': None, 'unknown_value': None})
    encoded, maps, unknown_codes = ordinal_encode_categories(x_cat)
    return (
        encoded,
        {
            'encoder': maps,
            'unknown_value': unknown_codes,
            'unknown_codes': unknown_codes,
        },
    )
