from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
import torch


SUBSAMPLE_LIMIT = 40_000
# The train context must fit a single model forward; rows*features of the train
# part is capped by this budget (val/test are never subsampled, features never).
TRAIN_CELL_BUDGET = 3_840_000
SUBSAMPLE_SEED = 0
_SUBSAMPLE_PART_OFFSETS = {"train": 0, "val": 1, "test": 2, "trainval": 3}


@dataclass
class TabularDataset:
    x: dict[str, pd.DataFrame]
    y: dict[str, np.ndarray]
    y_raw: dict[str, np.ndarray]
    task_type: str = "regression"
    score_name: str = "rmse"

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


def get_task_info(dataset_root: Path) -> tuple[str, str]:
    info_path = dataset_root / "info.json"
    if not info_path.exists():
        return "regression", "rmse"
    info = load_json(info_path)
    task = info.get("task")
    if not isinstance(task, dict):
        return "regression", "rmse"
    task_type = task.get("type") or "regression"
    score_name = task.get("score") or (
        "rmse" if task_type == "regression" else "accuracy"
    )
    if task_type not in {"regression", "binclass", "multiclass"}:
        raise RuntimeError(
            f"Dataset {dataset_root} has unsupported task.type={task_type!r}."
        )
    return task_type, score_name


def validate_regression_task(dataset_root: Path) -> None:
    task_type, _ = get_task_info(dataset_root)
    if task_type != "regression":
        raise RuntimeError(
            f"Dataset {dataset_root} has task.type={task_type!r}; this eval path is "
            "regression-only."
        )


CLASSIFICATION_TASKS = {"binclass", "multiclass", "classification"}


def normalize_task_type(task_type: str | None) -> str:
    if task_type in {None, "regression"}:
        return "regression"
    if task_type in CLASSIFICATION_TASKS:
        return "classification"
    raise ValueError(f"Unsupported task_type={task_type!r}")


def is_classification_task(task_type: str | None) -> bool:
    return normalize_task_type(task_type) == "classification"


def _load_or_create_subsample_indices(
    dataset_root: Path,
    split: str,
    part: str,
    indices: np.ndarray,
    *,
    limit: int = SUBSAMPLE_LIMIT,
    seed: int = SUBSAMPLE_SEED,
) -> np.ndarray:
    if indices.shape[0] <= limit:
        return indices
    cache_dir = dataset_root / "subsample" / f"limit{limit}_seed{seed}" / split
    cache_path = cache_dir / f"{part}.npy"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (limit,):
            return cached
    seed_seq = np.random.SeedSequence([seed, _SUBSAMPLE_PART_OFFSETS[part]])
    rng = np.random.default_rng(seed_seq)
    chosen = np.sort(rng.choice(indices.shape[0], size=limit, replace=False))
    subsampled = indices[chosen]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, subsampled)
    return subsampled


def load_split_indices(
    dataset_root: Path,
    split: str,
    *,
    train_limit: int = SUBSAMPLE_LIMIT,
    refit_context: bool = False,
) -> dict[str, np.ndarray]:
    split_dir = dataset_root / "splits" / split
    if not split_dir.exists():
        raise RuntimeError(f"Unknown split: {split_dir}")
    indices = {
        part: np.load(split_dir / f"{part}.npy") for part in ("train", "val", "test")
    }
    # Only the in-context support set is ever subsampled. In refit mode the
    # support set is drawn from the concatenated train+val pool (the tuned
    # recipe is frozen, so val may join the context); val/test parts stay the
    # untouched real splits either way.
    if refit_context:
        pool = np.concatenate([indices["train"], indices["val"]])
        indices["train"] = _load_or_create_subsample_indices(
            dataset_root, split, "trainval", pool, limit=train_limit
        )
    else:
        indices["train"] = _load_or_create_subsample_indices(
            dataset_root, split, "train", indices["train"], limit=train_limit
        )
    return indices


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


def _n_cols(array: np.ndarray | None) -> int:
    if array is None:
        return 0
    return int(array.shape[1]) if array.ndim > 1 else 1


def load_tabular_dataset(
    dataset_root: Path, split: str, *, refit_context: bool = False
) -> TabularDataset:
    dataset_root = resolve_dataset_root(dataset_root)
    task_type, score_name = get_task_info(dataset_root)

    x_num = load_optional_array(dataset_root, "x_num.npy")
    x_cat = load_optional_array(dataset_root, "x_cat.npy")
    x_bin = load_optional_array(dataset_root, "x_bin.npy")
    n_features = _n_cols(x_num) + _n_cols(x_cat) + _n_cols(x_bin)
    train_limit = min(
        SUBSAMPLE_LIMIT, max(1, TRAIN_CELL_BUDGET // max(1, n_features))
    )
    split_idx = load_split_indices(
        dataset_root, split, train_limit=train_limit, refit_context=refit_context
    )
    y_all = np.load(dataset_root / "y.npy")

    x = {
        part: _make_feature_frame(
            None if x_num is None else x_num[indices],
            None if x_cat is None else x_cat[indices],
            None if x_bin is None else x_bin[indices],
        )
        for part, indices in split_idx.items()
    }
    y_split = apply_split(y_all, split_idx)
    if task_type == "regression":
        y_raw = {
            part: values.astype(np.float32, copy=False).reshape(-1)
            for part, values in y_split.items()
        }
    else:
        y_raw = {part: values.reshape(-1) for part, values in y_split.items()}
    return TabularDataset(
        x=x,
        y=y_raw,
        y_raw=y_raw,
        task_type=task_type,
        score_name=score_name,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    rmse = float(sklearn.metrics.mean_squared_error(y_true, preds) ** 0.5)
    mae = float(sklearn.metrics.mean_absolute_error(y_true, preds))
    r2 = float(sklearn.metrics.r2_score(y_true, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true, proba, n_classes: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    labels = np.arange(n_classes)
    cross_entropy = float(sklearn.metrics.log_loss(y_true, proba, labels=labels))
    metrics = {
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, pred)),
        "log_loss": cross_entropy,
        "cross_entropy": cross_entropy,
        "cross-entropy": cross_entropy,
    }
    if n_classes == 2:
        try:
            auc = float(sklearn.metrics.roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            auc = float("nan")
        metrics["roc_auc"] = auc
        metrics["roc-auc"] = auc
    return metrics
