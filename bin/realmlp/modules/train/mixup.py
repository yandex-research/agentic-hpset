"""RealMLP training loop (v2): input mixup on the continuous block.

Hypothesis: v0's regularizers (dropout, weight decay, label smoothing) were
meta-tuned for the average benchmark dataset; on small-n overfit-prone
datasets, data-space augmentation is a stronger lever. Mixup (Zhang et al.
2018) is part of the regularization cocktail that made plain MLPs
state-of-the-art in Kadra et al. 2021 ("well-tuned simple nets"), a named
tabular result. The per-dataset tuner switches it on where it helps; on large
clean data it smooths real decision boundaries and is deselected.

Mechanism: identical to the v0 loop, except each training batch is mixed with
a within-batch shuffled copy at ``lam ~ Beta(alpha, alpha)`` (``mixup_alpha``,
default 0.2), with ``lam`` folded to ``[0.5, 1]`` so the original sample is
always the dominant side. The continuous block (numeric + one-hot + frequency
columns) is mixed linearly; large-cat embedding indices cannot be interpolated,
so the dominant side's indices are kept — a documented deviation from
published mixup that matters only where large-cat features dominate. The loss
is the standard mixup combination ``lam * loss(y) + (1 - lam) * loss(y_perm)``
computed through the unmodified loss-stage interface, so any loss module
composes.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn

from .._schedules import schedule_value
from ..optimizer import OptimizerBundle
from .realmlp import TrainMemberResult, _auto_batch_size


def train_member_mixup(
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
    mixup_alpha = float(cfg.get("mixup_alpha", 0.2))

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
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam = max(lam, 1.0 - lam)  # original sample is the dominant side
            pair = torch.randperm(idx.shape[1], device=device)
            idx_pair = idx[:, pair]

            optimizer.zero_grad(set_to_none=True)
            x_cont = lam * train_x_cont[idx] + (1.0 - lam) * train_x_cont[idx_pair]
            logits = model(x_cont, train_x_cat[idx])
            loss = lam * loss_fn(logits, train_y[idx], cfg, t)
            if lam < 1.0:
                loss = loss + (1.0 - lam) * loss_fn(logits, train_y[idx_pair], cfg, t)
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


__all__ = ["train_member_mixup"]
