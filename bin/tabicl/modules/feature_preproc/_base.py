from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class FeatureEncoder:
    def fit(self, x):
        if not hasattr(x, "columns"):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        self.columns_ = list(x.columns) if hasattr(x, "columns") else None
        if hasattr(x, "columns"):
            cat_cols = make_column_selector(
                dtype_include=["string", "object", "category", "boolean"]
            )(x)
            num_cols = make_column_selector(dtype_include="number")(x)
            self.tfm_ = ColumnTransformer(
                [
                    (
                        "categorical",
                        OrdinalEncoder(
                            dtype=np.int64,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                        cat_cols,
                    ),
                    ("continuous", SimpleImputer(), num_cols),
                ]
            )
        else:
            self.tfm_ = (
                OrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                if x.dtype.kind in {"b", "O", "S", "U"}
                else SimpleImputer()
            )
        transformed = self.tfm_.fit_transform(x).astype(np.float32)
        self.keep_ = (
            np.ones(transformed.shape[1], dtype=bool)
            if transformed.shape[0] <= 1
            else np.array(
                [
                    len(np.unique(transformed[:, index])) > 1
                    for index in range(transformed.shape[1])
                ]
            )
        )
        return self

    def transform(self, x) -> np.ndarray:
        if hasattr(x, "columns") and self.columns_ is not None:
            x = x[self.columns_]
        elif not hasattr(x, "columns"):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        return self.tfm_.transform(x).astype(np.float32)[:, self.keep_]

    def fit_transform(self, x) -> np.ndarray:
        return self.fit(x).transform(x)


# A normalizer factory receives the (post-standardize) array `z` and the random
# state and returns a fitted-on-demand sklearn transformer, or None for the
# identity case. `z` is passed so data-dependent params (e.g. QuantileTransformer
# n_quantiles) can be sized at fit time.
NormalizerFactory = Callable[[np.ndarray, "int | None"], object]


@dataclass
class FeaturePreprocessor:
    make_normalizer: NormalizerFactory
    standardize: bool = True
    outlier_threshold: float = 4.0
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> "FeaturePreprocessor":
        x = np.asarray(x, dtype=np.float32)
        if self.standardize:
            self.mean_ = x.mean(axis=0, keepdims=True)
            self.std_ = x.std(axis=0, keepdims=True) + 1e-6
            z = np.clip((x - self.mean_) / self.std_, -100, 100)
        else:
            self.mean_ = np.zeros((1, x.shape[1]), dtype=np.float32)
            self.std_ = np.ones((1, x.shape[1]), dtype=np.float32)
            z = x
        self.normalizer_ = self.make_normalizer(z, self.random_state)
        if self.normalizer_ is not None:
            self.min_, self.max_ = z.min(axis=0, keepdims=True), z.max(
                axis=0, keepdims=True
            )
            z = self.normalizer_.fit_transform(z)
        clean = z.copy()
        mean = np.nanmean(clean, axis=0)
        std = np.maximum(
            np.nanstd(clean, axis=0, ddof=1 if len(clean) > 1 else 0), 1e-6
        )
        clean[
            (clean < mean - self.outlier_threshold * std)
            | (clean > mean + self.outlier_threshold * std)
        ] = np.nan
        self.out_mean_ = np.nanmean(clean, axis=0)
        self.out_std_ = np.maximum(
            np.nanstd(clean, axis=0, ddof=1 if len(clean) > 1 else 0), 1e-6
        )
        self.lo_ = self.out_mean_ - self.outlier_threshold * self.out_std_
        self.hi_ = self.out_mean_ + self.outlier_threshold * self.out_std_
        self.x_train_ = self.transform(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.standardize:
            z = np.clip((x - self.mean_) / self.std_, -100, 100)
        else:
            z = x
        if self.normalizer_ is not None:
            try:
                z = self.normalizer_.transform(z)
            except ValueError:
                z = self.normalizer_.transform(np.clip(z, self.min_, self.max_))
        transformed = np.minimum(
            np.log1p(np.abs(z)) + self.hi_,
            np.maximum(-np.log1p(np.abs(z)) + self.lo_, z),
        )
        return transformed.astype(np.float32)


__all__ = ["FeatureEncoder", "FeaturePreprocessor", "NormalizerFactory"]
