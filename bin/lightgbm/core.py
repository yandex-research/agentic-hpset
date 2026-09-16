from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics


@dataclass
class TabularDataset:
    x: dict[str, pd.DataFrame]
    y: dict[str, np.ndarray]
    y_raw: dict[str, np.ndarray]

    @property
    def n_features(self) -> int:
        return int(self.x["train"].shape[1])

    def size(self, part: str) -> int:
        return int(self.y[part].shape[0])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def is_dataset_dir(path: Path) -> bool:
    feature_files = ("x_num.npy", "x_cat.npy", "x_bin.npy")
    return (
        path.is_dir()
        and (path / "y.npy").exists()
        and (path / "splits").is_dir()
        and any((path / name).exists() for name in feature_files)
    )


def resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser()
    if is_dataset_dir(path):
        return path
    if not path.exists():
        raise RuntimeError(f"Dataset path does not exist: {path}")
    candidates = sorted(
        candidate for candidate in path.rglob("*") if is_dataset_dir(candidate)
    )
    if not candidates:
        raise RuntimeError(
            f"No dataset directories found under {path}. Expected a directory with "
            "y.npy, splits/, and at least one of x_num.npy, x_cat.npy, or x_bin.npy."
        )
    if len(candidates) > 1:
        preview = ", ".join(str(candidate) for candidate in candidates[:8])
        suffix = " ..." if len(candidates) > 8 else ""
        raise RuntimeError(
            f"Ambiguous dataset path {path}: found {len(candidates)} datasets "
            f"({preview}{suffix}). Please pass an exact dataset directory."
        )
    return candidates[0]


def load_split_indices(dataset_root: Path, split: str) -> dict[str, np.ndarray]:
    split_dir = dataset_root / "splits" / split
    if not split_dir.exists():
        raise RuntimeError(f"Unknown split: {split_dir}")
    return {
        part: np.load(split_dir / f"{part}.npy") for part in ("train", "val", "test")
    }


def apply_split(
    array: np.ndarray, split: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {part: array[idx] for part, idx in split.items()}


def load_optional_array(dataset_root: Path, filename: str) -> np.ndarray | None:
    path = dataset_root / filename
    if not path.exists():
        return None
    return np.load(path)


def _as_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _make_numeric_frame(array: np.ndarray | None, prefix: str) -> pd.DataFrame:
    if array is None:
        return pd.DataFrame()
    values = _as_2d(array).astype(np.float32, copy=False)
    return pd.DataFrame(
        values,
        columns=[f"{prefix}_{index}" for index in range(values.shape[1])],
    )


def _make_categorical_frame(array: np.ndarray | None, prefix: str) -> pd.DataFrame:
    if array is None:
        return pd.DataFrame()
    values = _as_2d(array)
    columns = {}
    for index in range(values.shape[1]):
        columns[f"{prefix}_{index}"] = pd.Series(pd.Categorical(values[:, index]))
    return pd.DataFrame(columns)


def _make_feature_frame(
    x_num: np.ndarray | None,
    x_cat: np.ndarray | None,
    x_bin: np.ndarray | None,
) -> pd.DataFrame:
    parts = [
        _make_numeric_frame(x_num, "num"),
        _make_categorical_frame(x_cat, "cat"),
        _make_categorical_frame(x_bin, "bin"),
    ]
    frame = pd.concat(parts, axis=1)
    if frame.shape[1] == 0:
        raise RuntimeError("Dataset has no usable features.")
    return frame


def load_tabular_dataset(dataset_root: Path, split: str) -> TabularDataset:
    dataset_root = resolve_dataset_root(dataset_root)
    split_idx = load_split_indices(dataset_root, split)

    x_num = load_optional_array(dataset_root, "x_num.npy")
    x_cat = load_optional_array(dataset_root, "x_cat.npy")
    x_bin = load_optional_array(dataset_root, "x_bin.npy")
    y_all = np.load(dataset_root / "y.npy")

    x = {
        part: _make_feature_frame(
            None if x_num is None else x_num[indices],
            None if x_cat is None else x_cat[indices],
            None if x_bin is None else x_bin[indices],
        )
        for part, indices in split_idx.items()
    }
    y_raw = {
        part: values.astype(np.float32, copy=False).reshape(-1)
        for part, values in apply_split(y_all, split_idx).items()
    }
    return TabularDataset(x=x, y=y_raw, y_raw=y_raw)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


DEFAULT_FIT_SUBSAMPLE = 100_000


def subsample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    """Return indices for a random subsample, or all indices if n <= max_samples.

    Used by feature/data_aug modules to bound the cost of fitting transformers
    on large datasets (e.g., PCA / KMeans / NearestNeighbors over millions of
    rows). The transform path remains over the full data; only the fit is
    subsampled.
    """
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_samples), replace=False)


SCORE_TO_METRIC_KEY: dict[str, str] = {
    "rmse": "rmse",
    "mae": "mae",
    "accuracy": "accuracy",
    "roc-auc": "roc_auc",
    "cross-entropy": "log_loss",
}


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    rmse = float(sklearn.metrics.mean_squared_error(y_true, preds) ** 0.5)
    mae = float(sklearn.metrics.mean_absolute_error(y_true, preds))
    r2 = float(sklearn.metrics.r2_score(y_true, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(
    y_true: np.ndarray,
    y_pred_label: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.dtype.kind == "f":
        y_true = y_true.astype(np.int64)
    y_pred_label = np.asarray(y_pred_label).reshape(-1)
    metrics: dict[str, float] = {
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, y_pred_label)),
    }
    if y_pred_proba is None:
        return metrics
    proba = np.asarray(y_pred_proba)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    labels = list(range(proba.shape[1]))
    try:
        metrics["log_loss"] = float(sklearn.metrics.log_loss(y_true, proba, labels=labels))
    except ValueError:
        metrics["log_loss"] = float("nan")
    if proba.shape[1] == 2:
        try:
            metrics["roc_auc"] = float(sklearn.metrics.roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics
