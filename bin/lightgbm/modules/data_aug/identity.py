from __future__ import annotations

import numpy as np
import pandas as pd


def identity_aug(
    x_train: pd.DataFrame, y_train: np.ndarray, seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    return x_train.copy(), np.asarray(y_train).copy()
