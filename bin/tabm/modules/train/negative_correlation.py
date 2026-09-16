from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bin.tabm.core import Dataset, TorchDataset, maybe_sync, reset_gpu_stats, seed_everything

from .train import (
    InferenceFn,
    LossFn,
    OptimizerBuilder,
    _batch_tensors,
    _copy_state,
    _evaluate,
    _print_epoch,
    _report,
)


def _binclass_kl_diversity(preds: Tensor) -> Tensor:
    """Mean KL(P_head || mean_P) for Bernoulli heads with shape (batch, k)."""
    if preds.shape[-1] <= 1:
        return preds.new_zeros(())
    log_k = math.log(preds.shape[-1])
    log_p_pos = F.logsigmoid(preds)
    log_p_neg = F.logsigmoid(-preds)
    log_mean_pos = torch.logsumexp(log_p_pos, dim=-1, keepdim=True) - log_k
    log_mean_neg = torch.logsumexp(log_p_neg, dim=-1, keepdim=True) - log_k
    kl = log_p_pos.exp() * (log_p_pos - log_mean_pos) + log_p_neg.exp() * (
        log_p_neg - log_mean_neg
    )
    return kl.mean()


def _multiclass_kl_diversity(preds: Tensor) -> Tensor:
    """Mean KL(P_head || mean_P) for categorical heads with shape (batch, k, n_classes)."""
    if preds.shape[1] <= 1:
        return preds.new_zeros(())
    log_k = math.log(preds.shape[1])
    log_probs = F.log_softmax(preds, dim=-1)
    log_mean = torch.logsumexp(log_probs, dim=1, keepdim=True) - log_k
    kl = (log_probs.exp() * (log_probs - log_mean)).sum(dim=-1)
    return kl.mean()


def _ncl_loss(
    preds: Tensor,
    target: Tensor,
    dataset_meta: Dataset,
    loss_fn: LossFn,
    ncl_lambda: float,
) -> Tensor:
    if dataset_meta.is_regression:
        if preds.ndim != 2:
            return loss_fn(preds, target, dataset_meta)
        target_expanded = target.unsqueeze(-1).expand_as(preds)
        base_loss = loss_fn(preds, target_expanded, dataset_meta)
        pred_mean = preds.mean(dim=-1, keepdim=True)
        diversity = (preds - pred_mean).pow(2).mean()
        return base_loss - ncl_lambda * diversity

    if dataset_meta.is_binclass:
        if preds.ndim != 2:
            return loss_fn(preds, target, dataset_meta)
        target_expanded = target.unsqueeze(-1).expand_as(preds)
        base_loss = loss_fn(preds, target_expanded, dataset_meta)
        diversity = _binclass_kl_diversity(preds)
        return base_loss - ncl_lambda * diversity

    if dataset_meta.is_multiclass:
        if preds.ndim != 3:
            return loss_fn(preds, target, dataset_meta)
        target_expanded = target.unsqueeze(-1).expand(-1, preds.shape[1])
        base_loss = loss_fn(preds, target_expanded, dataset_meta)
        diversity = _multiclass_kl_diversity(preds)
        return base_loss - ncl_lambda * diversity

    raise RuntimeError(
        f"NCL training does not support task_type={dataset_meta.task_type!r}."
    )


def train_v6(
    model: nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    inference_fn: InferenceFn,
    loss_fn: LossFn,
    optimizer_builder: OptimizerBuilder,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    seed_everything(seed)
    model_name = getattr(model, "model_name", "tabm")
    optimizer = optimizer_builder(model, trial_params)
    batch_size = int(trial_params["batch_size"])
    gradient_clip = float(trial_params["gradient_clip"])
    ncl_lambda = float(trial_params.get("ncl_lambda", 0.1))
    train_size = dataset_meta.size("train")
    score_name = dataset_meta.score_name
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    last_it_s = 0
    reset_gpu_stats(device)
    maybe_sync(device)
    start = time.perf_counter()

    for epoch in range(1, int(trial_params["max_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_start = time.perf_counter()

        # NCL compares heads on the same rows, so unlike baseline TabM training
        # this loop uses one shared minibatch for every ensemble member.
        permutation = torch.randperm(train_size, device=device)
        for batch_idx in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            x_num, x_cat, target = _batch_tensors(dataset_tensors, batch_idx)
            loss = _ncl_loss(
                model(x_num, x_cat), target, dataset_meta, loss_fn, ncl_lambda
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite TabM loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            metrics, _ = _evaluate(
                model,
                dataset_tensors,
                dataset_meta,
                trial_params,
                inference_fn,
                device,
                ("val",),
            )
        val_score = float(metrics["val"]["score"])
        improved = val_score < best_score
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_state = _copy_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed = time.perf_counter() - start
        epoch_time = max(time.perf_counter() - epoch_start, 1e-6)
        last_it_s = max(1, math.trunc(math.ceil(train_size / batch_size) / epoch_time))
        _print_epoch(
            improved=improved,
            model_name=model_name,
            epoch=epoch,
            score_name=score_name,
            metrics=metrics,
            val_score=val_score,
            loss=float(np.mean(epoch_losses)),
            elapsed=elapsed,
            it_s=last_it_s,
        )
        if epochs_without_improvement >= int(trial_params["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        metrics, _ = _evaluate(
            model,
            dataset_tensors,
            dataset_meta,
            trial_params,
            inference_fn,
            device,
            ("train", "val", "test"),
        )
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    return (
        _report(
            model=model,
            model_name=model_name,
            dataset_meta=dataset_meta,
            device=device,
            best_epoch=best_epoch,
            elapsed=elapsed,
            it_s=last_it_s,
            metrics=metrics,
            extras={"ncl_lambda": ncl_lambda},
        ),
        best_state,
    )


__all__ = ["train_v6"]
