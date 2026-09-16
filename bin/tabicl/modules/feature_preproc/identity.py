from __future__ import annotations

from ._base import FeaturePreprocessor


def _make_identity(z, random_state):
    return None


def feature_preproc_identity(
    *, outlier_threshold: float = 4.0, random_state: int | None = None
) -> FeaturePreprocessor:
    return FeaturePreprocessor(
        make_normalizer=_make_identity,
        standardize=True,
        outlier_threshold=outlier_threshold,
        random_state=random_state,
    )


__all__ = ["feature_preproc_identity"]
