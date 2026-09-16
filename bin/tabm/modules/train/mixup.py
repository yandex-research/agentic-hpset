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
    _iter_batches,
    _print_epoch,
    _report,
)


def _mix_batch(
    x_num: Tensor | None,
    x_cat: Tensor | None,
    y: Tensor,
    alpha: float,
    generator: torch.Generator,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    batch_size = y.shape[0]
    perm = torch.randperm(batch_size, generator=generator, device=y.device)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    x_num_mixed = None if x_num is None else lam * x_num + (1.0 - lam) * x_num[perm]
    if x_cat is None:
        x_cat_mixed = None
    else:
        keep_left = torch.rand(batch_size, generator=generator, device=y.device) < lam
        view_shape = (batch_size,) + (1,) * (x_cat.ndim - 1)
        x_cat_mixed = torch.where(keep_left.view(view_shape), x_cat, x_cat[perm])
    y_mixed = lam * y + (1.0 - lam) * y[perm]
    return x_num_mixed, x_cat_mixed, y_mixed


def _soft_classification_loss(
    preds: Tensor,
    targets: Tensor,
    dataset_meta: Dataset,
) -> Tensor:
    if dataset_meta.is_binclass:
        return F.binary_cross_entropy_with_logits(preds, targets.float())
    log_probs = F.log_softmax(preds, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def _mix_classification_batch(
    x_num: Tensor | None,
    x_cat: Tensor | None,
    targets: Tensor,
    dataset_meta: Dataset,
    alpha: float,
    generator: torch.Generator,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    batch_size = targets.shape[0]
    perm = torch.randperm(batch_size, generator=generator, device=targets.device)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    x_num_mixed = None if x_num is None else lam * x_num + (1.0 - lam) * x_num[perm]
    if x_cat is None:
        x_cat_mixed = None
    else:
        keep_left = torch.rand(batch_size, generator=generator, device=targets.device) < lam
        x_cat_mixed = torch.where(
            keep_left.view((batch_size,) + (1,) * (x_cat.ndim - 1)),
            x_cat,
            x_cat[perm],
        )
    if dataset_meta.is_binclass:
        mixed_targets = lam * targets.float() + (1.0 - lam) * targets[perm].float()
    else:
        one_hot = F.one_hot(targets, int(dataset_meta.n_classes)).float()
        mixed_targets = lam * one_hot + (1.0 - lam) * one_hot[perm]
    return x_num_mixed, x_cat_mixed, mixed_targets


def train_v3(
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
    rng_gen = torch.Generator(device=device).manual_seed(int(seed))
    model_name = getattr(model, "model_name", "tabm")
    ensemble_size = int(getattr(model, "k", 1))
    optimizer = optimizer_builder(model, trial_params)
    batch_size = int(trial_params["batch_size"])
    gradient_clip = float(trial_params["gradient_clip"])
    mixup_alpha = float(trial_params.get("mixup_alpha", 0.2))
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
            if dataset_meta.is_regression:
                x_num, x_cat, targets = _mix_batch(
                    x_num, x_cat, targets, mixup_alpha, rng_gen
                )
                loss = loss_fn(model(x_num, x_cat), targets, dataset_meta)
            else:
                x_num, x_cat, targets = _mix_classification_batch(
                    x_num, x_cat, targets, dataset_meta, mixup_alpha, rng_gen
                )
                loss = _soft_classification_loss(model(x_num, x_cat), targets, dataset_meta)
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
        ),
        best_state,
    )


__all__ = ["train_v3"]
