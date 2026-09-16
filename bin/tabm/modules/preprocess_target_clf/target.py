from __future__ import annotations

from typing import Any

import numpy as np


def target_preprocess_clf_v0(
    y_raw: dict[str, np.ndarray],
    _config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    return y_raw, {"inverse_transform": None}
