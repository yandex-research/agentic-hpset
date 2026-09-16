from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bin.tabm.core import Dataset, TorchDataset, maybe_sync, reset_gpu_stats, seed_everything

from .mixup import _soft_classification_loss
from .train import (
    InferenceFn,
    LossFn,
    OptimizerBuilder,
    _batch_tensors,
    _copy_state,
    _evaluate,
    _iter_batches,
    _print_epoch,
    _report,
)


def _cutmix_batch(
    x_num: Tensor | None,
    x_cat: Tensor | None,
    targets: Tensor,
    dataset_meta: Dataset,
    cutmix_p: float,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    batch_size = targets.shape[0]
    permutation = torch.randperm(batch_size, device=targets.device)
    keep_ratio: Tensor = torch.ones(batch_size, device=targets.device)
    if cutmix_p > 0.0:
        ratios: list[Tensor] = []
        if x_num is not None:
            keep_mask_num = (
                torch.rand(batch_size, x_num.shape[-1], device=x_num.device) >= cutmix_p
            ).float()
            view_shape = (batch_size,) + (1,) * (x_num.ndim - 2) + (x_num.shape[-1],)
            x_num = keep_mask_num.view(view_shape) * x_num + (
                1.0 - keep_mask_num.view(view_shape)
            ) * x_num[permutation]
            ratios.append(keep_mask_num.mean(dim=1))
        if x_cat is not None:
            keep_mask_cat = torch.rand(batch_size, x_cat.shape[-1], device=x_cat.device) >= cutmix_p
            view_shape = (batch_size,) + (1,) * (x_cat.ndim - 2) + (x_cat.shape[-1],)
            x_cat = torch.where(keep_mask_cat.view(view_shape), x_cat, x_cat[permutation])
            ratios.append(keep_mask_cat.float().mean(dim=1))
        if ratios:
            keep_ratio = torch.stack(ratios).mean(dim=0)
    if dataset_meta.is_regression:
        while keep_ratio.ndim < targets.ndim:
            keep_ratio = keep_ratio.unsqueeze(-1)
        mixed_targets = keep_ratio * targets + (1.0 - keep_ratio) * targets[permutation]
    elif dataset_meta.is_binclass:
        while keep_ratio.ndim < targets.ndim:
            keep_ratio = keep_ratio.unsqueeze(-1)
        mixed_targets = keep_ratio * targets.float() + (1.0 - keep_ratio) * targets[permutation].float()
    else:
        one_hot = F.one_hot(targets, int(dataset_meta.n_classes)).float()
        while keep_ratio.ndim < one_hot.ndim:
            keep_ratio = keep_ratio.unsqueeze(-1)
        mixed_targets = keep_ratio * one_hot + (1.0 - keep_ratio) * one_hot[permutation]
    return x_num, x_cat, mixed_targets


def train_v8(
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
    ensemble_size = int(getattr(model, "k", 1))
    optimizer = optimizer_builder(model, trial_params)
    batch_size = int(trial_params["batch_size"])
    gradient_clip = float(trial_params["gradient_clip"])
    cutmix_p = float(trial_params.get("cutmix_p", 0.3))
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
        for batch_idx in _iter_batches(train_size, ensemble_size, batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            x_num, x_cat, targets = _batch_tensors(dataset_tensors, batch_idx)
            x_num, x_cat, targets = _cutmix_batch(
                x_num, x_cat, targets, dataset_meta, cutmix_p
            )
            preds = model(x_num, x_cat)
            loss = (
                loss_fn(preds, targets, dataset_meta)
                if dataset_meta.is_regression
                else _soft_classification_loss(preds, targets, dataset_meta)
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
            extras={"cutmix_p": cutmix_p},
        ),
        best_state,
    )


__all__ = ["train_v8"]
