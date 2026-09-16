"""``TabPreprocessor`` — orchestrator that chains the per-stage numerical and
categorical preprocessors selected from their registries.

Takes already-built stage dicts (from ``NUM_PREPROCESS_MAP[idx](cfg)`` and
``CAT_PREPROCESS_MAP[idx](cfg)``) so this class has no knowledge of the registry
layer. The agentic wrapper is responsible for resolving indices to builders.

Output: ``(x_cont, x_cat)`` where ``x_cont`` is the concatenation of the scaled
continuous block and the scaled small-cat one-hot block, and ``x_cat`` is the
int64 indices of the large-cardinality categorical features.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


CatIndicator = Union[List[bool], np.ndarray]


class TabPreprocessor:
    def __init__(
        self,
        num_stage: Dict[str, Callable[..., Any]],
        cat_stage: Dict[str, Callable[..., Any]],
    ) -> None:
        self._num = num_stage
        self._cat = cat_stage

    def fit(
        self,
        X_df: pd.DataFrame,
        *,
        cat_indicator: Optional[CatIndicator] = None,
        cat_col_names: Optional[List[str]] = None,
        stat_indices: Optional[np.ndarray] = None,
    ) -> "TabPreprocessor":
        self.columns_ = list(X_df.columns)
        X_stat = X_df if stat_indices is None else X_df.iloc[np.asarray(stat_indices, dtype=np.int64)]

        if cat_col_names is not None:
            if cat_indicator is not None:
                raise ValueError("Specify only one of cat_indicator and cat_col_names")
            cat_indicator = [col in cat_col_names for col in self.columns_]
        if cat_indicator is None:
            cat_indicator = [
                str(dtype) in ("object", "category", "string", "bool") or pd.api.types.is_bool_dtype(dtype)
                for dtype in X_df.dtypes
            ]
        if len(cat_indicator) != len(self.columns_):
            raise ValueError("cat_indicator length must match the number of columns")

        self.cat_cols_: List[Any] = [c for c, is_cat in zip(self.columns_, cat_indicator) if is_cat]
        self.cont_cols_: List[Any] = [c for c, is_cat in zip(self.columns_, cat_indicator) if not is_cat]

        cat_stat_df = X_stat[self.cat_cols_] if self.cat_cols_ else pd.DataFrame(index=X_stat.index)
        self.cat_artifacts_ = self._cat["fit"](cat_stat_df)

        cont_stat = (
            X_stat[self.cont_cols_].to_numpy(dtype=np.float32)
            if self.cont_cols_
            else np.zeros((len(X_stat), 0), dtype=np.float32)
        )
        self.cont_artifacts_ = self._num["fit"](cont_stat)

        one_hot_stat = self._cat["encode_small_onehot"](cat_stat_df, self.cat_artifacts_)
        self.onehot_artifacts_ = self._num["fit"](one_hot_stat)

        self.cat_sizes_: List[int] = self.cat_artifacts_["cat_sizes"]
        self.large_cat_sizes_: List[int] = self.cat_artifacts_["large_sizes"]
        self.n_num_ = len(self.cont_cols_)
        self.n_one_hot_ = one_hot_stat.shape[1]
        self.n_cont_ = self.n_num_ + self.n_one_hot_
        return self

    def transform(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["columns_", "cat_cols_", "cont_cols_"])
        X_df = X_df.reindex(columns=self.columns_)

        cont = (
            X_df[self.cont_cols_].to_numpy(dtype=np.float32)
            if self.cont_cols_
            else np.zeros((len(X_df), 0), dtype=np.float32)
        )
        cont = self._num["transform"](cont, self.cont_artifacts_)

        cat_df = X_df[self.cat_cols_] if self.cat_cols_ else pd.DataFrame(index=X_df.index)
        one_hot = self._cat["encode_small_onehot"](cat_df, self.cat_artifacts_)
        one_hot = self._num["transform"](one_hot, self.onehot_artifacts_)

        x_cont = np.concatenate([cont, one_hot], axis=1).astype(np.float32)
        x_cat = self._cat["encode_large_indices"](cat_df, self.cat_artifacts_)
        return x_cont, x_cat


__all__ = ["TabPreprocessor"]
