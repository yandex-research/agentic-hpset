from __future__ import annotations

from sklearn.preprocessing import PowerTransformer

from ._base import FeaturePreprocessor


def _make_power(z, random_state):
    return PowerTransformer(method="yeo-johnson", standardize=True)


def feature_preproc_power(
    *, outlier_threshold: float = 4.0, random_state: int | None = None
) -> FeaturePreprocessor:
    return FeaturePreprocessor(
        make_normalizer=_make_power,
        standardize=True,
        outlier_threshold=outlier_threshold,
        random_state=random_state,
    )


__all__ = ["feature_preproc_power"]
