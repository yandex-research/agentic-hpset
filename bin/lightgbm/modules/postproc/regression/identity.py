from __future__ import annotations

import numpy as np


def identity_postproc(
    preds: dict[str, np.ndarray],
    y_train: np.ndarray,
    task_type: str,
) -> dict[str, np.ndarray]:
    return {part: np.asarray(arr).copy() for part, arr in preds.items()}
