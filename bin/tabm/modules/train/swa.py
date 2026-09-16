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


@torch.no_grad()
def _swa_update(
    swa_state: dict[str, Tensor], model_state: dict[str, Tensor], n_avg: int
) -> None:
    for key, current in model_state.items():
        if key not in swa_state:
            swa_state[key] = current.detach().clone()
        else:
            running = swa_state[key]
            if running.dtype.is_floating_point:
                running.add_(current.detach() - running, alpha=1.0 / (n_avg + 1))
            else:
                running.copy_(current.detach())


def train_v2(
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
    max_epochs = int(trial_params["max_epochs"])
    swa_start_epoch = max(1, int(0.5 * max_epochs))
    train_size = dataset_meta.size("train")
    score_name = dataset_meta.score_name
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    last_it_s = 0
    swa_state: dict[str, Tensor] = {}
    n_avg = 0
    reset_gpu_stats(device)
    maybe_sync(device)
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_start = time.perf_counter()
        for batch_idx in _iter_batches(train_size, ensemble_size, batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            x_num, x_cat, targets = _batch_tensors(dataset_tensors, batch_idx)
            loss = loss_fn(model(x_num, x_cat), targets, dataset_meta)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite TabM loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        if epoch >= swa_start_epoch:
            _swa_update(swa_state, model.state_dict(), n_avg)
            n_avg += 1

        live_backup: dict[str, Tensor] | None = None
        if swa_state:
            live_backup = _copy_state(model)
            model.load_state_dict(swa_state)
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
        if live_backup is not None:
            model.load_state_dict(live_backup)

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
            suffix=f" [swa_n] {n_avg}",
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
            extras={"swa_n_averaged": int(n_avg)},
        ),
        best_state,
    )


__all__ = ["train_v2"]
