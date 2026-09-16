from __future__ import annotations

from sklearn.preprocessing import QuantileTransformer

from ._base import FeaturePreprocessor


def _make_quantile_uniform(z, random_state):
    n_quantiles = max(2, min(1000, z.shape[0]))
    return QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="uniform",
        random_state=random_state,
        subsample=10**9,
    )


def feature_preproc_quantile_uniform(
    *, outlier_threshold: float = 4.0, random_state: int | None = None
) -> FeaturePreprocessor:
    return FeaturePreprocessor(
        make_normalizer=_make_quantile_uniform,
        standardize=True,
        outlier_threshold=outlier_threshold,
        random_state=random_state,
    )


__all__ = ["feature_preproc_quantile_uniform"]
