"""LightGBM regressor/classifier + HP search space.

The search space mirrors the requested LightGBM grid. Regression targets are
standardized by the training wrapper, matching the previous base behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


LIGHTGBM_FIXED_PARAMS: dict[str, Any] = {
    "n_estimators": 4000,
    "early_stopping_rounds": 200,
    "n_jobs": 16,
    "verbosity": -1,
    # LightGBM ignores bagging_fraction unless bagging_freq is positive.
    "bagging_freq": 1,
}


N_TRIALS = 200
N_STARTUP_TRIALS = 20


def _is_categorical_like(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def prepare_lightgbm_frames(
    *frames: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, ...], list[str]]:
    """Copy frames and normalize categorical columns for LightGBM.

    LightGBM's sklearn API can consume pandas categoricals directly. Some
    augmentation variants may turn categoricals into object/string columns, so
    we detect those too and cast them back to categorical dtype. Categories are
    learned from the training frame; unseen validation/test values become
    missing values instead of leaking evaluation-set vocabularies into training.
    """
    if not frames:
        return (), []

    out = tuple(frame.copy() for frame in frames)
    train = out[0]
    categorical_cols: list[str] = []
    for name in train.columns:
        if any(_is_categorical_like(frame[name]) for frame in out):
            train_values = train[name].astype("string").fillna("__nan__")
            categories = pd.Index(pd.unique(train_values))
            if "__nan__" not in categories:
                categories = categories.append(pd.Index(["__nan__"]))
            for frame in out:
                values = frame[name].astype("string").fillna("__nan__")
                frame[name] = pd.Categorical(values, categories=categories)
            categorical_cols.append(str(name))
        else:
            for frame in out:
                frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(
                    np.float32
                )
    return out, categorical_cols


def resolve_lightgbm_search_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Translate Optuna's conditional params into LightGBM constructor params."""
    resolved = dict(params or {})
    lambda_l2_mode = resolved.pop("lambda_l2_mode", None)
    if lambda_l2_mode == "zero":
        resolved["lambda_l2"] = 0.0
    elif lambda_l2_mode not in (None, "loguniform"):
        raise ValueError(f"Unsupported lambda_l2_mode: {lambda_l2_mode!r}")
    return resolved


def split_lightgbm_params(
    params: dict[str, Any] | None,
    *,
    seed: int,
    thread_count: int | None = None,
) -> tuple[dict[str, Any], int | None]:
    """Return model constructor params and fit-time early stopping rounds."""
    fixed_params = dict(LIGHTGBM_FIXED_PARAMS)
    if thread_count is not None:
        fixed_params["n_jobs"] = thread_count

    merged = {
        **fixed_params,
        **resolve_lightgbm_search_params(params),
        "random_state": seed,
    }
    early_stopping_rounds = merged.pop("early_stopping_rounds", None)
    return merged, (
        None if early_stopping_rounds is None else int(early_stopping_rounds)
    )


def get_lightgbm_early_stopping_rounds(
    params: dict[str, Any] | None = None,
) -> int | None:
    value = {**LIGHTGBM_FIXED_PARAMS, **(params or {})}.get("early_stopping_rounds")
    return None if value is None else int(value)


def get_lightgbm_eval_metric(task_type: str, metric_key: str | None = None) -> str:
    if task_type == "regression":
        return "rmse"
    if task_type == "binclass":
        if metric_key == "roc_auc":
            return "auc"
        if metric_key == "accuracy":
            return "binary_error"
        return "binary_logloss"
    if task_type == "multiclass":
        if metric_key == "accuracy":
            return "multi_error"
        return "multi_logloss"
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def suggest_lightgbm_params(trial) -> dict[str, Any]:
    """Requested LightGBM search space. `trial` is an `optuna.Trial`."""
    lambda_l2_mode = trial.suggest_categorical("lambda_l2_mode", ["zero", "loguniform"])
    params: dict[str, Any] = {
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "lambda_l2_mode": lambda_l2_mode,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 768),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-4, 100.0, log=True
        ),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }
    if lambda_l2_mode == "loguniform":
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 0.1, 10.0, log=True)
    else:
        params["lambda_l2"] = 0.0
    return params


def build_lightgbm(
    task_type: str,
    params: dict[str, Any] | None,
    seed: int,
    thread_count: int | None = None,
) -> LGBMRegressor | LGBMClassifier:
    """Instantiate LightGBM for regression, binary, or multiclass tasks."""
    merged, _early_stopping_rounds = split_lightgbm_params(
        params, seed=seed, thread_count=thread_count
    )
    if task_type == "regression":
        return LGBMRegressor(objective="regression", **merged)
    if task_type == "binclass":
        return LGBMClassifier(objective="binary", **merged)
    if task_type == "multiclass":
        return LGBMClassifier(objective="multiclass", **merged)
    raise ValueError(f"Unsupported task_type: {task_type!r}")
