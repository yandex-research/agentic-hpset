from __future__ import annotations

import pandas as pd


def identity_features(
    x: dict[str, pd.DataFrame], y_train,
) -> dict[str, pd.DataFrame]:
    return {part: frame.copy() for part, frame in x.items()}
