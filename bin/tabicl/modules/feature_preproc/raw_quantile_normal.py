from __future__ import annotations

from sklearn.preprocessing import QuantileTransformer

from ._base import FeaturePreprocessor


def _make_raw_quantile_normal(z, random_state):
    n_quantiles = max(2, min(1000, z.shape[0]))
    return QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=random_state,
        subsample=10**9,
    )


def feature_preproc_raw_quantile_normal(
    *, outlier_threshold: float = 4.0, random_state: int | None = None
) -> FeaturePreprocessor:
    # `standardize=False`: skip the (x-mean)/std clip in BOTH fit and transform so
    # the quantile transform sees raw feature values (and transform does not clip
    # to +/-100). This is the load-bearing difference from quantile_normal.
    return FeaturePreprocessor(
        make_normalizer=_make_raw_quantile_normal,
        standardize=False,
        outlier_threshold=outlier_threshold,
        random_state=random_state,
    )


__all__ = ["feature_preproc_raw_quantile_normal"]
