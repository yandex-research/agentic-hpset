from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.special
import sklearn.metrics
import torch
from torch import Tensor


@dataclass
class Dataset:
    x_num: dict[str, np.ndarray] | None
    x_cat: dict[str, np.ndarray] | None
    y: dict[str, np.ndarray]
    y_raw: dict[str, np.ndarray]
    task_type: str
    score_name: str
    n_classes: int
    preprocess_artifacts: dict[str, dict[str, object]]

    @property
    def n_num_features(self) -> int:
        return 0 if self.x_num is None else int(self.x_num["train"].shape[1])

    @property
    def cat_cardinalities(self) -> list[int]:
        if self.x_cat is None or self.x_cat["train"].shape[1] == 0:
            return []
        artifact_cardinalities = self.preprocess_artifacts.get("categorical", {}).get(
            "cardinalities"
        )
        if artifact_cardinalities is not None:
            cardinalities = [int(value) for value in artifact_cardinalities]
            if len(cardinalities) != self.x_cat["train"].shape[1]:
                raise RuntimeError(
                    "Categorical cardinality metadata length does not match "
                    f"x_cat feature count: {len(cardinalities)} vs "
                    f"{self.x_cat['train'].shape[1]}."
                )
            if any(value < 1 for value in cardinalities):
                raise RuntimeError(
                    f"Categorical cardinalities must be positive, got {cardinalities}."
                )
            return cardinalities
        return [
            int(self.x_cat["train"][:, i].max()) + 1
            for i in range(self.x_cat["train"].shape[1])
        ]

    def size(self, part: str) -> int:
        return int(self.y[part].shape[0])

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    @property
    def is_binclass(self) -> bool:
        return self.task_type == "binclass"

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == "multiclass"


@dataclass
class TorchDataset:
    x_num: dict[str, Tensor] | None
    x_cat: dict[str, Tensor] | None
    y: dict[str, Tensor]

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
            f"No dataset directories found under {path}. "
            "Expected a directory with y.npy, splits/, and at least one of "
            "x_num.npy, x_cat.npy, or x_bin.npy."
        )
    if len(candidates) > 1:
        preview = ", ".join(str(candidate) for candidate in candidates[:8])
        suffix = " ..." if len(candidates) > 8 else ""
        raise RuntimeError(
            f"Ambiguous dataset path {path}: found {len(candidates)} datasets "
            f"({preview}{suffix}). Please pass an exact dataset directory."
        )
    return candidates[0]


def infer_dataset_name(data_root: Path) -> str:
    parts = data_root.resolve().parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "data":
            return "/".join(parts[index + 1 :])
    return data_root.name


def load_task_info(dataset_root: Path) -> tuple[str, str]:
    info_path = dataset_root / "info.json"
    if not info_path.exists():
        raise RuntimeError(f"Missing dataset metadata: {info_path}")
    info = load_json(info_path)
    task = info.get("task")
    if not isinstance(task, dict):
        raise RuntimeError(f"Dataset info is missing the 'task' object: {info_path}")
    task_type = task.get("type")
    score_name = task.get("score")
    if task_type == "regression":
        if score_name != "rmse":
            raise RuntimeError(
                f"Unsupported regression score {score_name!r} in {info_path}. "
                "Expected: rmse."
            )
        return task_type, score_name
    if task_type not in {"binclass", "multiclass"}:
        raise RuntimeError(
            f"Unsupported task.type={task_type!r} in {info_path}. "
            "Expected one of: regression, binclass, multiclass."
        )
    if score_name not in {"accuracy", "cross-entropy", "roc-auc"}:
        raise RuntimeError(
            f"Unsupported classification score {score_name!r} in {info_path}. "
            "Expected one of: accuracy, cross-entropy, roc-auc."
        )
    if task_type == "multiclass" and score_name == "roc-auc":
        raise RuntimeError("roc-auc is supported only for binary classification datasets.")
    return task_type, score_name


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


def load_optional_split_array(
    dataset_root: Path,
    filename: str,
    split: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    path = dataset_root / filename
    if not path.exists():
        return None
    return apply_split(np.load(path), split)


def convert_bin_to_cat(
    x_bin: dict[str, np.ndarray] | None,
    x_cat: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    if x_bin is None:
        return x_cat
    x_bin_as_cat = {
        part: np.where(np.isnan(values), 2.0, values).astype(np.int64)
        for part, values in x_bin.items()
    }
    if x_cat is None:
        return x_bin_as_cat
    return {
        part: np.column_stack([x_cat[part], x_bin_as_cat[part]])
        for part in x_cat
    }


def validate_and_prepare_labels(
    y_raw: dict[str, np.ndarray],
    task_type: str,
) -> dict[str, np.ndarray]:
    all_labels = np.concatenate([np.asarray(values).reshape(-1) for values in y_raw.values()])
    rounded = np.round(all_labels)
    if not np.allclose(all_labels, rounded):
        raise RuntimeError("Classification labels must be integer-valued.")
    unique_labels = np.unique(rounded.astype(np.int64))
    expected_labels = np.arange(len(unique_labels), dtype=np.int64)
    if not np.array_equal(unique_labels, expected_labels):
        raise RuntimeError(
            f"Classification labels must be zero-based and contiguous, got "
            f"{unique_labels.tolist()}."
        )
    prepared = {
        part: np.asarray(values).reshape(-1).astype(np.int64)
        for part, values in y_raw.items()
    }
    n_classes = int(len(unique_labels))
    if task_type == "binclass" and n_classes != 2:
        raise RuntimeError(
            f"Binary classification dataset must have exactly 2 classes, got {n_classes}."
        )
    if task_type == "multiclass" and n_classes < 3:
        raise RuntimeError(
            f"Multiclass dataset must have at least 3 classes, got {n_classes}."
        )
    return prepared


def load_raw_arrays(
    dataset_root: Path,
    split: str,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray],
    str,
    str,
    int,
]:
    dataset_root = resolve_dataset_root(Path(dataset_root))
    task_type, score_name = load_task_info(dataset_root)
    split_idx = load_split_indices(dataset_root, split)
    x_num = load_optional_split_array(dataset_root, "x_num.npy", split_idx)
    x_cat = load_optional_split_array(dataset_root, "x_cat.npy", split_idx)
    x_bin = load_optional_split_array(dataset_root, "x_bin.npy", split_idx)
    raw_y = apply_split(np.load(dataset_root / "y.npy"), split_idx)
    if task_type == "regression":
        y_raw = {
            part: np.asarray(values).reshape(-1).astype(np.float32)
            for part, values in raw_y.items()
        }
        n_classes = 0
    else:
        y_raw = validate_and_prepare_labels(raw_y, task_type)
        n_classes = int(len(np.unique(np.concatenate([y_raw[part] for part in y_raw]))))
    return (
        x_num,
        convert_bin_to_cat(x_bin, x_cat),
        y_raw,
        task_type,
        score_name,
        n_classes,
    )


def to_torch(dataset: Dataset, device: torch.device) -> TorchDataset:
    x_num = (
        None
        if dataset.x_num is None
        else {
            part: torch.as_tensor(values, dtype=torch.float32, device=device)
            for part, values in dataset.x_num.items()
        }
    )
    x_cat = (
        None
        if dataset.x_cat is None
        else {
            part: torch.as_tensor(values, dtype=torch.long, device=device)
            for part, values in dataset.x_cat.items()
        }
    )
    y_dtype = torch.long if dataset.is_multiclass else torch.float32
    y = {
        part: torch.as_tensor(values, dtype=y_dtype, device=device)
        for part, values in dataset.y.items()
    }
    return TorchDataset(x_num=x_num, x_cat=x_cat, y=y)


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
    y_true = np.asarray(y_true).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    rmse = float(sklearn.metrics.mean_squared_error(y_true, preds) ** 0.5)
    mae = float(sklearn.metrics.mean_absolute_error(y_true, preds))
    r2 = float(sklearn.metrics.r2_score(y_true, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2, "score": rmse}


def logits_to_probs(logits: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "binclass":
        return scipy.special.expit(np.asarray(logits).reshape(-1))
    return scipy.special.softmax(logits, axis=1)


def classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    task_type: str,
    score_name: str,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    if task_type == "binclass":
        probs = logits_to_probs(logits, task_type)
        labels = (probs >= 0.5).astype(np.int64)
        metrics = {
            "accuracy": float(sklearn.metrics.accuracy_score(y_true, labels)),
            "cross-entropy": float(sklearn.metrics.log_loss(y_true, probs, labels=[0, 1])),
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc-auc"] = float(sklearn.metrics.roc_auc_score(y_true, probs))
    else:
        probs = logits_to_probs(logits, task_type)
        labels = probs.argmax(axis=1).astype(np.int64)
        metrics = {
            "accuracy": float(sklearn.metrics.accuracy_score(y_true, labels)),
            "cross-entropy": float(
                sklearn.metrics.log_loss(y_true, probs, labels=np.arange(probs.shape[1]))
            ),
        }
    higher_is_better = score_name in {"accuracy", "roc-auc"}
    score_value = metrics[score_name]
    metrics["score"] = -score_value if higher_is_better else score_value
    return metrics


def compute_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    task_type: str,
    score_name: str,
) -> dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, preds)
    return classification_metrics(y_true, preds, task_type, score_name)


def compute_metrics_for_parts(
    dataset_meta: Dataset,
    predictions: dict[str, np.ndarray],
    parts: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    return {
        part: compute_metrics(
            dataset_meta.y_raw[part],
            predictions[part],
            dataset_meta.task_type,
            dataset_meta.score_name,
        )
        for part in parts
    }


def capture_gpu_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    return {"peak_allocated_mb": float(allocated), "peak_reserved_mb": float(reserved)}


def reset_gpu_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
