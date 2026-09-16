"""Runtime data types and metric helpers for one self-contained MLP package."""

from __future__ import annotations

import random
from dataclasses import dataclass

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
        return 0 if self.x_num is None else int(self.x_num['train'].shape[1])

    @property
    def cat_cardinalities(self) -> list[int]:
        if self.x_cat is None or self.x_cat['train'].shape[1] == 0:
            return []
        return [
            int(self.x_cat['train'][:, i].max()) + 2
            for i in range(self.x_cat['train'].shape[1])
        ]

    def size(self, part: str) -> int:
        return int(self.y[part].shape[0])

    @property
    def is_regression(self) -> bool:
        return self.task_type == 'regression'

    @property
    def is_binclass(self) -> bool:
        return self.task_type == 'binclass'

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == 'multiclass'


@dataclass
class TorchDataset:
    x_num: dict[str, Tensor] | None
    x_cat: dict[str, Tensor] | None
    y: dict[str, Tensor]


def bounded_numerical_parts(
    x_num: dict[str, np.ndarray], *, tail_quantile: float = 0.005
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Impute and clip every split using finite training-only support."""
    if not 0.0 <= tail_quantile < 0.5:
        raise ValueError(f'Invalid tail quantile: {tail_quantile}')
    train = np.asarray(x_num['train'], dtype=np.float64)
    if train.ndim != 2:
        raise ValueError(f'Expected a 2D numerical matrix, got {train.shape}')
    fill = np.empty(train.shape[1], dtype=np.float64)
    lower = np.empty_like(fill)
    upper = np.empty_like(fill)
    for column in range(train.shape[1]):
        finite = train[:, column][np.isfinite(train[:, column])]
        if finite.size == 0:
            fill[column] = lower[column] = upper[column] = 0.0
            continue
        fill[column] = np.median(finite)
        lower[column], upper[column] = np.quantile(
            finite, [tail_quantile, 1.0 - tail_quantile]
        )

    bounded: dict[str, np.ndarray] = {}
    for part, values in x_num.items():
        array = np.asarray(values, dtype=np.float64)
        array = np.where(np.isnan(array), fill, array)
        array = np.where(np.isposinf(array), upper, array)
        array = np.where(np.isneginf(array), lower, array)
        bounded[part] = np.clip(array, lower, upper)
    return bounded, {'fill': fill, 'lower': lower, 'upper': upper}


def categorical_key(value: object) -> str:
    """Return a deterministic, type-preserving key for arbitrary categories."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return '<MISSING>'
    try:
        if bool(np.isnan(value)):
            return '<MISSING>'
    except (TypeError, ValueError):
        pass
    value_type = type(value)
    return f'{value_type.__module__}.{value_type.__qualname__}:{value!r}'


def ordinal_encode_categories(
    x_cat: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, int]], np.ndarray]:
    """Fit deterministic category maps on train and reserve one unknown code."""
    train = np.asarray(x_cat['train'])
    maps: list[dict[str, int]] = []
    for column in range(train.shape[1]):
        keys = sorted({categorical_key(value) for value in train[:, column]})
        maps.append({key: index for index, key in enumerate(keys)})
    unknown_codes = np.asarray([len(mapping) for mapping in maps], dtype=np.int64)
    encoded: dict[str, np.ndarray] = {}
    for part, values in x_cat.items():
        output = np.empty(np.asarray(values).shape, dtype=np.int64)
        for column, mapping in enumerate(maps):
            unknown = int(unknown_codes[column])
            output[:, column] = [
                mapping.get(categorical_key(value), unknown)
                for value in np.asarray(values)[:, column]
            ]
        encoded[part] = output
    return encoded, maps, unknown_codes


def validate_and_prepare_labels(
    y_raw: dict[str, np.ndarray], task_type: str
) -> dict[str, np.ndarray]:
    all_labels = np.concatenate(
        [np.asarray(values).reshape(-1) for values in y_raw.values()]
    )
    rounded = np.round(all_labels)
    if not np.allclose(all_labels, rounded):
        raise RuntimeError('Classification labels must be integer-valued.')
    unique_labels = np.unique(rounded.astype(np.int64))
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        raise RuntimeError(
            'Classification labels must be zero-based and contiguous, got '
            f'{unique_labels.tolist()}.'
        )
    n_classes = len(unique_labels)
    if task_type == 'binclass' and n_classes != 2:
        raise RuntimeError(
            f'Binary classification requires 2 classes, got {n_classes}.'
        )
    if task_type == 'multiclass' and n_classes < 3:
        raise RuntimeError(
            f'Multiclass classification requires >=3 classes, got {n_classes}.'
        )
    return {
        part: np.asarray(values).reshape(-1).astype(np.int64)
        for part, values in y_raw.items()
    }


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


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    rmse = float(sklearn.metrics.mean_squared_error(y_true, preds) ** 0.5)
    return {
        'rmse': rmse,
        'mae': float(sklearn.metrics.mean_absolute_error(y_true, preds)),
        'r2': float(sklearn.metrics.r2_score(y_true, preds)),
        'score': rmse,
    }


def classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    task_type: str,
    score_name: str,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    if task_type == 'binclass':
        probs = scipy.special.expit(np.asarray(logits).reshape(-1))
        labels = (probs >= 0.5).astype(np.int64)
        metrics = {
            'accuracy': float(sklearn.metrics.accuracy_score(y_true, labels)),
            'cross-entropy': float(
                sklearn.metrics.log_loss(y_true, probs, labels=[0, 1])
            ),
        }
        if len(np.unique(y_true)) == 2:
            metrics['roc-auc'] = float(sklearn.metrics.roc_auc_score(y_true, probs))
    else:
        probs = scipy.special.softmax(np.asarray(logits), axis=1)
        labels = probs.argmax(axis=1).astype(np.int64)
        metrics = {
            'accuracy': float(sklearn.metrics.accuracy_score(y_true, labels)),
            'cross-entropy': float(
                sklearn.metrics.log_loss(
                    y_true, probs, labels=np.arange(probs.shape[1])
                )
            ),
        }
    value = metrics[score_name]
    metrics['score'] = -value if score_name in {'accuracy', 'roc-auc'} else value
    return metrics


def compute_metrics(
    y_true: np.ndarray, preds: np.ndarray, task_type: str, score_name: str
) -> dict[str, float]:
    if task_type == 'regression':
        return regression_metrics(y_true, preds)
    return classification_metrics(y_true, preds, task_type, score_name)


def compute_metrics_for_parts(
    dataset: Dataset,
    predictions: dict[str, np.ndarray],
    parts: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    return {
        part: compute_metrics(
            dataset.y_raw[part],
            predictions[part],
            dataset.task_type,
            dataset.score_name,
        )
        for part in parts
    }


def capture_gpu_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != 'cuda':
        return {'peak_allocated_mb': None, 'peak_reserved_mb': None}
    torch.cuda.synchronize(device)
    return {
        'peak_allocated_mb': float(torch.cuda.max_memory_allocated(device) / 1024**2),
        'peak_reserved_mb': float(torch.cuda.max_memory_reserved(device) / 1024**2),
    }


def reset_gpu_stats(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def maybe_sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
