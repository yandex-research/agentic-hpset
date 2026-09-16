"""Assemble, fit, and evaluate a modular LightGBM configuration.

The pipeline composes three axes of registered implementations:
  - feature: feature engineering / selection on x
  - data_aug: train-time data augmentation
  - postproc: post-processing of LightGBM predictions
              (regression -> raw preds; classification -> proba)

The classification postproc registry differs from the regression one. Optuna
draws a postproc index from the registry that matches the dataset's task_type,
which avoids wasting trials on no-op samples.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from bin._pipeline import RunResult, dataset_path, split_name

from bin.lightgbm.modules.data_aug import DATA_AUG_MAP
from bin.lightgbm.modules.feature import FEATURE_MAP
from bin.lightgbm.modules.postproc import (
    get_postproc_map,
)


REGISTRY_KEY_MAP = {
    "feature_idx": "feature",
    "data_aug_idx": "data_aug",
    "postproc_idx": "postproc",
}

FEATURE_MODE_KEY = "feature_mode"
FEATURE_MODES = ("append", "replace")
DEFAULT_FEATURE_MODE = "append"
IMPLEMENTATION_CONFIG_KEYS = frozenset((*REGISTRY_KEY_MAP.keys(), FEATURE_MODE_KEY))


def _copy_frames(x: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {part: frame.copy() for part, frame in x.items()}


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    return left.reset_index(drop=True).equals(right.reset_index(drop=True))


def _unique_column_name(frame: pd.DataFrame, base_name: str) -> str:
    if base_name not in frame.columns:
        return base_name
    index = 1
    while f"{base_name}_{index}" in frame.columns:
        index += 1
    return f"{base_name}_{index}"


def _replacement_columns(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
) -> list[str]:
    cols: list[str] = []
    for col in transformed.columns:
        if col not in original.columns:
            cols.append(col)
            continue
        if not _series_equal(original[col], transformed[col]):
            cols.append(col)
    return cols


def normalize_feature_mode(mode: object | None) -> str:
    if mode is None:
        return DEFAULT_FEATURE_MODE
    mode_str = str(mode)
    if mode_str not in FEATURE_MODES:
        choices = ", ".join(FEATURE_MODES)
        raise ValueError(f"Unknown feature_mode {mode_str!r}. Available: [{choices}]")
    return mode_str


def normalize_implementation_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    for arg_name in REGISTRY_KEY_MAP:
        if arg_name in out:
            out[arg_name] = int(out[arg_name])
    out[FEATURE_MODE_KEY] = normalize_feature_mode(out.get(FEATURE_MODE_KEY))
    return out


def get_implementation_indices(config: dict[str, Any]) -> dict[str, int]:
    normalized = normalize_implementation_config(config)
    return {key: int(normalized[key]) for key in REGISTRY_KEY_MAP if key in normalized}


def get_registries(task_type: str) -> dict[str, dict[int, Any]]:
    return {
        "feature": FEATURE_MAP,
        "data_aug": DATA_AUG_MAP,
        "postproc": get_postproc_map(task_type),
    }


def resolve_impl(registry: dict[int, Any], idx: int, name: str) -> Any:
    if idx not in registry:
        choices = ", ".join(str(key) for key in sorted(registry))
        raise ValueError(f"Unknown {name} index {idx}. Available indices: [{choices}]")
    return registry[idx]


def validate_implementation_indices(
    implementation_indices: dict[str, Any], task_type: str
) -> None:
    implementation_indices = normalize_implementation_config(implementation_indices)
    registries = get_registries(task_type)
    for arg_name, registry_name in REGISTRY_KEY_MAP.items():
        if arg_name not in implementation_indices:
            raise ValueError(f"Missing implementation index: {arg_name}")
        resolve_impl(
            registries[registry_name], implementation_indices[arg_name], registry_name
        )


def sample_implementation_indices(
    trial: optuna.Trial, task_type: str
) -> dict[str, Any]:
    registries = get_registries(task_type)
    out: dict[str, Any] = {}
    for arg_name, registry_name in REGISTRY_KEY_MAP.items():
        choices = sorted(registries[registry_name].keys())
        out[arg_name] = int(trial.suggest_categorical(arg_name, choices))
    out[FEATURE_MODE_KEY] = trial.suggest_categorical(FEATURE_MODE_KEY, FEATURE_MODES)
    return out


def apply_feature(
    feature_idx: int,
    x: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    mode: str = DEFAULT_FEATURE_MODE,
) -> dict[str, pd.DataFrame]:
    mode = normalize_feature_mode(mode)
    fn = resolve_impl(FEATURE_MAP, feature_idx, "feature")
    transformed = fn(x, y_train)
    if mode == "replace":
        train_cols = _replacement_columns(x["train"], transformed["train"])
        if not train_cols:
            return _copy_frames(x)
        return {part: transformed[part][train_cols].copy() for part in x}

    suffix = f"__{getattr(fn, '__name__', 'feature')}"
    out: dict[str, pd.DataFrame] = {}
    for part, original in x.items():
        new_frame = original.copy()
        transformed_frame = transformed[part]
        if len(original) != len(transformed_frame):
            raise ValueError(
                f"Feature implementation {feature_idx} changed row count for {part}: "
                f"{len(original)} -> {len(transformed_frame)}"
            )
        for col in transformed_frame.columns:
            transformed_col = transformed_frame[col].reset_index(drop=True)
            if col not in original.columns:
                new_frame[col] = transformed_col.to_numpy(copy=False)
                if isinstance(transformed_col.dtype, pd.CategoricalDtype):
                    new_frame[col] = transformed_col.values
                continue
            if _series_equal(original[col], transformed_col):
                continue
            appended_name = _unique_column_name(new_frame, f"{col}{suffix}")
            new_frame[appended_name] = transformed_col.to_numpy(copy=False)
            if isinstance(transformed_col.dtype, pd.CategoricalDtype):
                new_frame[appended_name] = transformed_col.values
        out[part] = new_frame

    if all(out[part].shape[1] == x[part].shape[1] for part in x):
        return _copy_frames(x)
    return out


def apply_data_aug(
    data_aug_idx: int, x_train: pd.DataFrame, y_train: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    fn = resolve_impl(DATA_AUG_MAP, data_aug_idx, "data_aug")
    return fn(x_train, y_train, seed)


def apply_postproc(
    postproc_idx: int,
    *,
    task_type: str,
    preds: dict[str, np.ndarray],
    proba: dict[str, np.ndarray] | None,
    y_train: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Run the chosen postproc and return (labels-or-preds, proba) for downstream metrics.

    Regression: postproc operates on `preds` (the float regression output);
    `proba` is unused.

    Classification: postproc operates on `proba`; labels are re-derived as
    argmax(postprocessed_proba) so calibration changes flow through to both
    label-based and proba-based metrics.
    """
    registry = get_postproc_map(task_type)
    fn = resolve_impl(registry, postproc_idx, "postproc")
    if task_type == "regression":
        new_preds = fn(preds, y_train, task_type)
        return new_preds, None
    if proba is None:
        raise RuntimeError("Classification pipeline requires non-None proba.")
    new_proba = fn(proba, y_train, task_type)
    new_preds: dict[str, np.ndarray] = {}
    for part, arr in new_proba.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = np.column_stack([1.0 - arr, arr])
        new_preds[part] = np.argmax(arr, axis=1).astype(np.float32)
    return new_preds, new_proba


def run_full_pipeline(
    *, task_type: str, metric_key: str, lightgbm_params: dict[str, Any],
    impl_indices: dict[str, Any], seed: int,
    dataset_x: dict[str, pd.DataFrame], dataset_y: dict[str, np.ndarray],
    thread_count: int = 16, lightgbm_overrides: dict[str, Any] | None = None,
) -> RunResult:
    """Feature -> augmentation -> normalized-target fit -> raw-space postprocessing.

    This is the portable fit/evaluation core of the research tuning driver.
    Classification predictions are probabilities; regression predictions are in
    the original target units. The CPU LightGBM recipe uses validation early stopping.
    """
    import lightgbm as lgb
    from bin.lightgbm.core import classification_metrics, regression_metrics
    from bin.lightgbm.model import (
        build_lightgbm, get_lightgbm_early_stopping_rounds,
        get_lightgbm_eval_metric, prepare_lightgbm_frames,
    )

    started = time.perf_counter()
    impl = normalize_implementation_config(impl_indices)
    validate_implementation_indices(impl, task_type)
    x = apply_feature(impl["feature_idx"], dataset_x, dataset_y["train"], impl[FEATURE_MODE_KEY])
    x_aug, y_aug = apply_data_aug(impl["data_aug_idx"], x["train"], dataset_y["train"], seed)
    parts = tuple(x)
    frames, cats = prepare_lightgbm_frames(x_aug, *(x[part] for part in parts))
    x_fit, x_query = frames[0], dict(zip(parts, frames[1:], strict=True))

    y_mean, y_std = None, None
    if task_type == "regression":
        y_mean, y_std = float(y_aug.mean()), float(y_aug.std()) or 1.0
        y_fit = ((y_aug - y_mean) / y_std).astype(np.float32)
        y_val = ((dataset_y["val"] - y_mean) / y_std).astype(np.float32)
    else:
        y_fit = y_aug.astype(np.int64)
        y_val = dataset_y["val"].astype(np.int64)

    params = {**(lightgbm_overrides or {}), **lightgbm_params}
    model = build_lightgbm(task_type, params, seed=seed, thread_count=thread_count)
    callbacks = [lgb.log_evaluation(period=0)]
    patience = get_lightgbm_early_stopping_rounds(params)
    if patience is not None and patience > 0:
        callbacks.append(lgb.early_stopping(patience, first_metric_only=True, verbose=False))
    model.fit(
        x_fit, y_fit, eval_set=[(x_query["val"], y_val)], eval_names=["validation"],
        eval_metric=get_lightgbm_eval_metric(task_type, metric_key),
        categorical_feature=cats or "auto", callbacks=callbacks,
    )
    if task_type == "regression":
        preds = {
            part: (np.asarray(model.predict(frame)).reshape(-1) * y_std + y_mean).astype(np.float32)
            for part, frame in x_query.items()
        }
        proba = None
    else:
        proba = {part: np.asarray(model.predict_proba(frame)) for part, frame in x_query.items()}
        preds = {part: values.argmax(axis=1) for part, values in proba.items()}
    preds, proba = apply_postproc(
        impl["postproc_idx"], task_type=task_type, preds=preds, proba=proba,
        y_train=y_aug if task_type == "regression" else dataset_y["train"],
    )
    metrics = {
        part: regression_metrics(dataset_y[part], preds[part]) if task_type == "regression"
        else classification_metrics(dataset_y[part], preds[part], proba[part])
        for part in parts
    }
    predictions = preds if proba is None else proba
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Non-finite LightGBM predictions.")
    report = {
        "metrics": metrics, "metric_key": metric_key, "task_type": task_type,
        "prediction_type": "labels" if proba is None else "probs",
        "time_seconds": time.perf_counter() - started,
        "best_iteration": model.best_iteration_, "tree_count": model.n_iter_,
        "implementation_config": impl,
    }
    if y_mean is not None:
        report["y_standardized"] = {"mean": y_mean, "std": y_std}
    return RunResult(report, predictions, model)


def run(config: dict[str, Any], *, dataset_root: str | Path | None = None,
        device: str | None = None) -> RunResult:
    """Run a single-model report config or a recovered ensemble member config."""
    from bin.lightgbm.core import SCORE_TO_METRIC_KEY, load_json, load_tabular_dataset

    if device is not None and str(device) != "cpu":
        raise ValueError("The released LightGBM recipe uses device='cpu'.")
    root = dataset_path(config, dataset_root)
    task = load_json(root / "info.json")["task"]
    dataset = load_tabular_dataset(root, split_name(config))
    params = dict(config.get("trial_params", config.get("lightgbm_params", {})))
    seed = int(config.get("seed", config.get("fit_seed", params.get("seed", 0))))
    params.pop("seed", None)
    impl = {key: 0 for key in REGISTRY_KEY_MAP}
    impl.update({key: config[key] for key in IMPLEMENTATION_CONFIG_KEYS if key in config})
    impl.update(config.get("implementation_indices", {}))
    impl.update(config.get("implementation_config", {}))
    overrides = {key: config[key] for key in ("n_estimators", "early_stopping_rounds") if key in config}
    result = run_full_pipeline(
        task_type=task["type"], metric_key=SCORE_TO_METRIC_KEY[task["score"]],
        lightgbm_params=params, impl_indices=impl, seed=seed,
        dataset_x=dataset.x, dataset_y=dataset.y,
        thread_count=int(config.get("thread_count", 16)), lightgbm_overrides=overrides,
    )
    result.report["score_name"] = task["score"]
    return result
