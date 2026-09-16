"""RealMLP per-member training loop (v0).

Runs ``n_epochs`` over the train tensors. Each member of the ``n_ens`` bundle
sees its own random permutation, so the ensemble is trained jointly under a
single batched forward pass. The optimizer's ``update_fn`` is invoked at every
step to apply the lr/wd schedule and the manual lr-coupled weight decay
(``weight_decay_fn``) is applied **before** ``optimizer.step()``.

Validation scoring uses an injected ``score_fn`` (resolved from the inference
stage by the wrapper) so the train loop is identical for classification and
regression — only the scoring closure changes.

Early stopping fires when ``epoch > es_mult * best_epoch + es_add`` and
``use_early_stopping`` is set. Honors ``time_limit_s`` if provided.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from .._schedules import schedule_value
from ..optimizer import OptimizerBundle

def _auto_batch_size(n_train: int) -> int:
    if n_train <= 1024:
        return 64
    if n_train <= 8192:
        return 128
    if n_train <= 30_000:
        return 256
    if n_train <= 100_000:
        return 512
    return 1024

@dataclass
class TrainMemberResult:
    best_state: Dict[str, torch.Tensor]
    best_score: float
    best_epoch: int

def train_member_v0(
    model: nn.Module,
    optimizer_bundle: OptimizerBundle,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any], float], torch.Tensor],
    train_x_cont: torch.Tensor,
    train_x_cat: torch.Tensor,
    train_y: torch.Tensor,
    val_x_cont: torch.Tensor,
    val_x_cat: torch.Tensor,
    val_y: torch.Tensor,
    score_fn: Optional[Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], float]],
    cfg: Dict[str, Any],
    *,
    has_val: bool,
    time_limit_s: Optional[float] = None,
) -> TrainMemberResult:
    device = train_x_cont.device
    n_models = int(getattr(model, "n_models", 1))

    batch_size_cfg = cfg.get("batch_size", 256)
    if batch_size_cfg == "auto":
        batch_size_cfg = _auto_batch_size(train_x_cont.shape[0])
    batch_size = min(int(batch_size_cfg), train_x_cont.shape[0])
    n_batches = max(1, train_x_cont.shape[0] // batch_size)
    n_iter_samples = n_batches * batch_size
    n_epochs = int(cfg.get("n_epochs", 256))

    optimizer = optimizer_bundle.optimizer
    update_fn = optimizer_bundle.update_fn
    weight_decay_fn = optimizer_bundle.weight_decay_fn

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    completed_samples = 0
    start = time.time()

    for epoch in range(n_epochs):
        model.train()
        perms = torch.stack(
            [torch.randperm(train_x_cont.shape[0], device=device) for _ in range(n_models)],
            dim=0,
        )
        for start_idx in range(0, n_iter_samples, batch_size):
            t = completed_samples / max(1, n_epochs * n_iter_samples)
            update_fn(optimizer, cfg, t)
            model.set_dropout(
                float(cfg.get("p_drop", 0.0))
                * schedule_value(cfg.get("p_drop_sched", "flat_cos"), t)
            )
            idx = perms[:, start_idx:start_idx + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x_cont[idx], train_x_cat[idx])
            loss = loss_fn(logits, train_y[idx], cfg, t)
            loss.backward()
            weight_decay_fn(optimizer)
            optimizer.step()
            completed_samples += batch_size

        if has_val and score_fn is not None:
            score = score_fn(model, val_x_cont, val_x_cat, val_y)
            if score <= best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if cfg.get("use_early_stopping", False):
                patience = int(cfg.get("early_stopping_additive_patience", 20))
                mult = float(cfg.get("early_stopping_multiplicative_patience", 2))
                current_epoch = epoch + 1
                if current_epoch > best_epoch * mult + patience:
                    break
        if time_limit_s is not None and time.time() - start > float(time_limit_s):
            break

    return TrainMemberResult(best_state=best_state, best_score=best_score, best_epoch=best_epoch)

__all__ = [
    "TrainMemberResult",
    "train_member_v0",
    ]
