from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
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


def _sam_perturb(
    parameters: list[Tensor],
    grads: list[Tensor],
    rho: float,
) -> list[Tensor]:
    eps = 1e-12
    grad_norm_sq = torch.zeros((), device=parameters[0].device)
    for grad in grads:
        grad_norm_sq = grad_norm_sq + grad.pow(2).sum()
    scale = rho / grad_norm_sq.sqrt().clamp_min(eps)
    perturbations: list[Tensor] = []
    with torch.no_grad():
        for parameter, grad in zip(parameters, grads, strict=True):
            perturbation = grad * scale
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    return perturbations


def _sam_unperturb(parameters: list[Tensor], perturbations: list[Tensor]) -> None:
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations, strict=True):
            parameter.sub_(perturbation)


def train_v5(
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
    sam_rho = float(trial_params.get("sam_rho", 0.05))
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
            x_num, x_cat, targets = _batch_tensors(dataset_tensors, batch_idx)

            optimizer.zero_grad(set_to_none=True)
            loss_first = loss_fn(model(x_num, x_cat), targets, dataset_meta)
            if not torch.isfinite(loss_first):
                raise RuntimeError("Encountered non-finite TabM loss.")
            loss_first.backward()
            params_with_grad = [p for p in model.parameters() if p.grad is not None]
            if not params_with_grad:
                raise RuntimeError("SAM step did not produce gradients.")
            grads_first = [p.grad.detach().clone() for p in params_with_grad]
            perturbations = _sam_perturb(params_with_grad, grads_first, sam_rho)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_num, x_cat), targets, dataset_meta)
            if not torch.isfinite(loss):
                _sam_unperturb(params_with_grad, perturbations)
                raise RuntimeError("Encountered non-finite TabM loss.")
            loss.backward()
            _sam_unperturb(params_with_grad, perturbations)
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
            extras={"sam_rho": sam_rho},
        ),
        best_state,
    )


__all__ = ["train_v5"]
