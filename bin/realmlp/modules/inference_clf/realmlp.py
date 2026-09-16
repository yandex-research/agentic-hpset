"""RealMLP classification inference (v0).

- ``raw_member_predict`` is the shared primitive: transform ``X`` through the
  member's preprocessor, batch the result through the member's model on its
  current device, and return the ``(n_ens, total, out_dim)`` logit tensor on
  CPU. Re-used by the regression inference module.
- ``average_classification_logits`` collapses the ensemble dim into a single
  ``(batch, n_classes)`` logit matrix using either pre-softmax averaging
  (``ens_av_before_softmax=True``) or post-softmax averaging in log space.
- ``score_classifier`` returns either class error (default) or cross-entropy
  on the validation tensors, used inside the train loop.
- ``predict_proba_members_v0`` aggregates probabilities across ensemble members
  for the public ``predict_proba`` API.
- ``predict_logits_ensemble`` returns per-member softmax probabilities stacked
  along the model dim (used by ``predict_proba_ensemble``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger

_MIN_PREDICT_BATCH_SIZE = 1


def _is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _clear_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _forward_batches(
    model: torch.nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    bs: int,
    *,
    to_cpu: bool,
) -> torch.Tensor:
    """Chunked forward. Returns ``(n_ens, N, out_dim)`` concatenated along the batch dim."""
    n = x_cont.shape[0]
    if n <= bs and not to_cpu:
        return model(x_cont, x_cat)
    chunks: List[torch.Tensor] = []
    for start in range(0, n, bs):
        out = model(x_cont[start:start + bs], x_cat[start:start + bs])
        chunks.append(out.cpu() if to_cpu else out)
    return torch.cat(chunks, dim=1)


def _forward_with_oom_retry(
    model: torch.nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    to_cpu: bool = False,
) -> torch.Tensor:
    """Chunked forward that halves ``cfg['predict_batch_size']`` on CUDA OOM.

    Mirrors ``bin/realmlp/base/realmlp.py::_predict_with_oom_retry`` — mutates
    ``cfg`` in place so subsequent calls (e.g. later epochs of ``score_fn``)
    start from the already-shrunk size.
    """
    while True:
        bs = int(cfg["predict_batch_size"])
        try:
            return _forward_batches(model, x_cont, x_cat, bs, to_cpu=to_cpu)
        except Exception as err:
            if not _is_oom(err) or bs <= _MIN_PREDICT_BATCH_SIZE:
                raise
            next_bs = max(_MIN_PREDICT_BATCH_SIZE, bs // 2)
            logger.warning(
                f"forward OOM with predict_batch_size={bs};"
                f" retrying with predict_batch_size={next_bs}"
            )
            cfg["predict_batch_size"] = next_bs
            _clear_cache(x_cont.device)


def raw_member_predict(
    preprocessor: Any,
    model: torch.nn.Module,
    X: pd.DataFrame,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Transform + batched forward of a single member. Returns ``(n_ens, N, out_dim)`` on CPU."""
    x_cont, x_cat = preprocessor.transform(X)
    device = next(model.parameters()).device
    x_cont_t = torch.as_tensor(x_cont, dtype=torch.float32, device=device)
    x_cat_t = torch.as_tensor(x_cat, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        return _forward_with_oom_retry(model, x_cont_t, x_cat_t, cfg, to_cpu=True)

def average_classification_logits(logits: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """Collapse ``(n_ens, batch, n_classes)`` -> ``(batch, n_classes)``."""
    if logits.shape[0] == 1:
        return logits[0]
    if cfg.get("ens_av_before_softmax", False):
        return logits.mean(dim=0)
    return torch.log(torch.softmax(logits, dim=-1).mean(dim=0) + 1e-30)

def score_classifier(
    model: torch.nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    y: torch.Tensor,
    cfg: Dict[str, Any],
) -> float:
    model.eval()
    with torch.no_grad():
        raw = _forward_with_oom_retry(model, x_cont, x_cat, cfg)
        pred = average_classification_logits(raw, cfg)
        metric = cfg.get("val_metric_name", None) or "class_error"
        if metric == "cross_entropy":
            return float(F.cross_entropy(pred, y).item())
        return float((pred.argmax(dim=1) != y).float().mean().item())

def predict_proba_members_v0(
    members: List[Any],
    X: pd.DataFrame,
    params: Dict[str, Any],
) -> np.ndarray:
    """Average probabilities across members for ``predict_proba``."""
    probs: List[torch.Tensor] = []
    for member in members:
        logits = raw_member_predict(member.preprocessor, member.model, X, params)
        if params.get("ens_av_before_softmax", False):
            probs.append(torch.softmax(logits.mean(dim=0), dim=-1))
        else:
            probs.append(torch.softmax(logits, dim=-1).mean(dim=0))
    return torch.stack(probs, dim=0).mean(dim=0).numpy()

def predict_logits_ensemble(
    members: List[Any],
    X: pd.DataFrame,
    params: Dict[str, Any],
) -> np.ndarray:
    """Per-member softmax probabilities, stacked along the model dim.

    Returns a ``(sum_member_n_ens, batch, n_classes)`` numpy array. Used by the
    wrapper's ``predict_proba_ensemble`` to expose ensemble-level probabilities.
    """
    chunks: List[np.ndarray] = []
    for member in members:
        logits = raw_member_predict(member.preprocessor, member.model, X, params)
        chunks.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(chunks, axis=0)

__all__ = [
    "_forward_with_oom_retry",
    "raw_member_predict",
    "average_classification_logits",
    "score_classifier",
    "predict_proba_members_v0",
    "predict_logits_ensemble",
    ]
