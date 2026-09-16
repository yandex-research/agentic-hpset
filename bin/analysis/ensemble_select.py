"""Greedy (Caruana) ensemble selection over a pool of model predictions.

This is the path-free, reusable core of the ensembling used in the paper. Given a
library of per-config validation/test predictions, :func:`caruana_te` runs greedy
weighted selection (Caruana et al., 2004) on the validation predictions and
returns the selected weights + the blended val/test predictions.

The committed ensemble results in ``exp/ensembled`` were produced by running this
over raw per-config prediction libraries (many GB) that are **not** included in
this repo, so the selections here are readable and reusable but cannot be re-run
bit-for-bit. Read the committed results with :mod:`bin.analysis.ensemble_io`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

HIGHER_IS_BETTER = {"roc-auc", "roc_auc", "accuracy", "acc"}
LOWER_IS_BETTER = {"rmse", "cross-entropy", "log-loss", "log_loss"}


def is_higher_better(score_name: str) -> bool:
    score = score_name.lower()
    if score in HIGHER_IS_BETTER:
        return True
    if score in LOWER_IS_BETTER:
        return False
    raise ValueError(score_name)


def score_to_error(score: float, score_name: str) -> float:
    return float(1.0 - score) if is_higher_better(score_name) else float(score)


def score_to_signed_error(score: float, score_name: str) -> float:
    return float(-score) if is_higher_better(score_name) else float(score)


def detect_task_type(y_val: np.ndarray) -> str:
    if np.issubdtype(y_val.dtype, np.integer):
        return "multiclass" if len(np.unique(y_val)) > 2 else "binclass"
    return "regression"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def normalize_pred(pred: np.ndarray, task_type: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if task_type == "regression":
        return pred.reshape(-1)
    if task_type == "binclass":
        if pred.ndim == 2 and pred.shape[1] == 2:
            in_unit = pred.min() >= -1e-6 and pred.max() <= 1.0 + 1e-6
            if in_unit and np.allclose(pred.sum(axis=1), 1.0, atol=1e-3):
                return np.clip(pred, 0.0, 1.0)
            return softmax(pred)
        if pred.ndim == 1:
            p1 = np.clip(pred, 0.0, 1.0) if pred.min() >= -1e-6 and pred.max() <= 1.0 + 1e-6 else sigmoid(pred)
            return np.column_stack([1.0 - p1, p1])
    if task_type == "multiclass" and pred.ndim == 2:
        in_unit = pred.min() >= -1e-6 and pred.max() <= 1.0 + 1e-6
        if in_unit and np.allclose(pred.sum(axis=1), 1.0, atol=1e-3):
            return np.clip(pred, 0.0, 1.0)
        return softmax(pred)
    raise ValueError(f"Unexpected prediction shape {pred.shape} for {task_type}")


def _proba_clean(p: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return p / p.sum(axis=-1, keepdims=True)


def metric_score(y_true: np.ndarray, y_pred: np.ndarray, score_name: str) -> float:
    score = score_name.lower()
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)
    if score == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred.reshape(-1))))
    if score in {"cross-entropy", "log-loss", "log_loss"}:
        proba = _proba_clean(y_pred)
        return float(log_loss(y_true.astype(int), proba, labels=np.arange(proba.shape[1])))
    if score in {"roc-auc", "roc_auc"}:
        values = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
        return float(roc_auc_score(y_true.astype(int), values))
    if score in {"accuracy", "acc"}:
        labels = (y_pred >= 0.5).astype(int) if y_pred.ndim == 1 else y_pred.argmax(axis=1)
        return float(accuracy_score(y_true.astype(int), labels))
    raise ValueError(score)


def _binary_auc_many(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], np.nan)
    order = np.argsort(scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1)
    rank_sum_pos = ranks[:, y].sum(axis=1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def vectorized_errors(y_true: np.ndarray, preds: np.ndarray, score_name: str) -> np.ndarray:
    score = score_name.lower()
    if score == "rmse":
        diff = preds - y_true.reshape(1, -1)
        return np.sqrt((diff * diff).mean(axis=1))
    if score in {"cross-entropy", "log-loss", "log_loss"}:
        eps = np.finfo(np.float64).eps
        proba = np.clip(preds, eps, 1.0 - eps)
        proba = proba / proba.sum(axis=-1, keepdims=True)
        idx = np.arange(preds.shape[1])
        true = y_true.astype(int)
        picked = proba[:, idx, true]
        return -np.log(picked).mean(axis=1)
    if score in {"roc-auc", "roc_auc"}:
        scores = preds if preds.ndim == 2 else preds[:, :, 1]
        return 1.0 - _binary_auc_many(y_true, scores)
    if score in {"accuracy", "acc"}:
        labels = (preds >= 0.5).astype(int) if preds.ndim == 2 else preds.argmax(axis=2)
        return 1.0 - (labels == y_true.astype(int)).mean(axis=1)
    raise ValueError(score)


def selected_member_test_corr(test_preds: np.ndarray, weights: np.ndarray) -> float:
    selected = np.where(weights > 0)[0]
    if selected.size < 2:
        return float("nan")
    flat = test_preds[selected].reshape(selected.size, -1).astype(np.float64)
    flat = flat[flat.std(axis=1) > 0]
    if flat.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(flat)
    pairs = corr[np.triu_indices(corr.shape[0], k=1)]
    pairs = pairs[~np.isnan(pairs)]
    return float(pairs.mean()) if pairs.size else float("nan")


def caruana_te(val_preds: np.ndarray, test_preds: np.ndarray, y_val: np.ndarray,
               score_name: str, *, n_steps: int):
    """Greedy weighted ensemble selection on validation predictions.

    Returns ``(weights, trace, selected, blended_val, blended_test)`` where
    ``weights`` are the per-member selection frequencies and the blended
    predictions are their weighted average.
    """
    counts = np.zeros(val_preds.shape[0], dtype=np.int64)
    selected: list[int] = []
    trace: list[float] = []
    current_sum = np.zeros_like(val_preds[0], dtype=np.float64)
    for step in range(n_steps):
        candidates = (current_sum[None, ...] + val_preds) / float(step + 1)
        errors = vectorized_errors(y_val, candidates, score_name)
        best = int(np.nanargmin(errors))
        counts[best] += 1
        selected.append(best)
        trace.append(float(errors[best]))
        current_sum = current_sum + val_preds[best]
    weights = counts / float(counts.sum())
    return weights, trace, selected, np.tensordot(weights, val_preds, axes=(0, 0)), np.tensordot(weights, test_preds, axes=(0, 0))
