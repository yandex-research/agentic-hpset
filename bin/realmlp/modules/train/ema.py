"""RealMLP training loop (v1): EMA weight averaging for checkpoint selection.

Hypothesis: v0 selects a single best epoch by a noisy validation score — on
small validation sets that selection is high-variance, and no meta-tuned
default can smooth it. Maintaining an exponential moving average of the
weights (Polyak averaging; SWA line, Izmailov et al. 2018; standard EMA
practice across CV) and scoring/checkpointing the *averaged* weights smooths
both the trajectory and the selection. Harm case: EMA lags the multi-cycle
``coslog4`` schedule and can blur across restarts on short runs — the decay
default is gentle and the tuner deselects where it loses.

Mechanism: identical to the v0 loop (per-member shuffling, per-step
``update_fn``, decoupled weight decay before ``optimizer.step()``, early
stopping), plus an EMA of the ``state_dict`` updated once per epoch with
``ema_decay`` (default 0.85). Validation scoring temporarily loads the EMA
weights, and ``best_state`` holds the EMA weights of the best epoch, so the
fitted member returned to inference is the averaged model.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from .._schedules import schedule_value
from ..optimizer import OptimizerBundle
from .realmlp import TrainMemberResult, _auto_batch_size


def _ema_update(
    ema_state: Dict[str, torch.Tensor],
    current_state: Dict[str, torch.Tensor],
    decay: float,
) -> None:
    with torch.no_grad():
        for key, value in current_state.items():
            target = ema_state[key]
            if torch.is_floating_point(value):
                target.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                target.copy_(value)


def train_member_ema(
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
    ema_decay = float(cfg.get("ema_decay", 0.85))

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

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_state = copy.deepcopy(ema_state)
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

        _ema_update(ema_state, model.state_dict(), ema_decay)

        if has_val and score_fn is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            score = score_fn(model, val_x_cont, val_x_cat, val_y)
            model.load_state_dict(backup)
            if score <= best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(ema_state)
            if cfg.get("use_early_stopping", False):
                patience = int(cfg.get("early_stopping_additive_patience", 20))
                mult = float(cfg.get("early_stopping_multiplicative_patience", 2))
                current_epoch = epoch + 1
                if current_epoch > best_epoch * mult + patience:
                    break
        else:
            best_state = copy.deepcopy(ema_state)
        if time_limit_s is not None and time.time() - start > float(time_limit_s):
            break

    return TrainMemberResult(best_state=best_state, best_score=best_score, best_epoch=best_epoch)


__all__ = ["train_member_ema"]
