from __future__ import annotations

from typing import TypeVar

import numpy as np
from torch import Tensor


Prediction = TypeVar("Prediction", Tensor, np.ndarray)


def mean_regression_output(preds: Prediction) -> Prediction:
    if preds.ndim >= 2:
        return preds[..., 0]
    return preds


__all__ = ["mean_regression_output"]
