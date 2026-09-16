"""SwapNoise input augmentation for distribution-shift robustness.

On every training batch, each numerical and categorical cell is independently
replaced (with probability `p_swap`, default 0.15) by the value at the same
column from another row in the same batch. Breaks brittle column-wise
correlations and forces the network to rely on robust multivariate signal,
which transfers better to drifted val/test distributions. Single forward
pass per step.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from bin.mlp.core import (
    Dataset,
    TorchDataset,
    capture_gpu_stats,
    compute_metrics_for_parts,
    maybe_sync,
    reset_gpu_stats,
    seed_everything,
)


InferenceFn = Callable[
    [nn.Module, TorchDataset, Dataset, dict[str, Any], torch.device, tuple[str, ...]],
    dict[str, np.ndarray],
]
LossFn = Callable[[Tensor, Tensor, Dataset], Tensor]
OptimizerBuilder = Callable[[nn.Module, dict[str, Any]], torch.optim.Optimizer]


def _swap_noise(x: Tensor | None, p: float) -> Tensor | None:
    if x is None or x.shape[0] < 2 or p <= 0.0:
        return x
    n = x.shape[0]
    mask = torch.rand_like(x, dtype=torch.float32) < p
    perm = torch.randint(0, n, (n,), device=x.device)
    swapped = x[perm]
    return torch.where(mask, swapped, x)


def train_swap_noise_v0(
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
    optimizer = optimizer_builder(model, trial_params)
    batch_size = int(trial_params["batch_size"])
    gradient_clip = float(trial_params["gradient_clip"])
    p_swap = float(trial_params.get("swap_noise_p", 0.15))
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    train_size = dataset_meta.size("train")
    last_it_s = 0
    score_name = dataset_meta.score_name
    reset_gpu_stats(device)
    maybe_sync(device)
    start = time.perf_counter()
    for epoch in range(1, int(trial_params["max_epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        permutation = torch.randperm(train_size, device=device)
        epoch_start = time.perf_counter()
        for batch_idx in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            x_num_b = (
                None if dataset_tensors.x_num is None else dataset_tensors.x_num["train"][batch_idx]
            )
            x_cat_b = (
                None if dataset_tensors.x_cat is None else dataset_tensors.x_cat["train"][batch_idx]
            )
            x_num_b = _swap_noise(x_num_b, p_swap)
            x_cat_b = _swap_noise(x_cat_b, p_swap)
            preds = model(x_num_b, x_cat_b)
            targets = dataset_tensors.y["train"][batch_idx]
            loss = loss_fn(preds, targets, dataset_meta)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite MLP loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            predictions = inference_fn(
                model, dataset_tensors, dataset_meta, trial_params, device, ("val", "test")
            )
        if not all(np.isfinite(predictions[part]).all() for part in predictions):
            raise RuntimeError("Encountered non-finite MLP predictions.")
        metrics = compute_metrics_for_parts(dataset_meta, predictions, ("val", "test"))
        val_score = float(metrics["val"]["score"])
        improved = val_score < best_score
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - start
        epoch_time = max(time.perf_counter() - epoch_start, 1e-6)
        last_it_s = max(1, math.trunc(math.ceil(train_size / batch_size) / epoch_time))
        print(
            f"{'*' if improved else ' '} [mlp epoch] {epoch:<3}"
            f" [val {score_name}] {metrics['val'][score_name]:.4f}"
            f" [test {score_name}] {metrics['test'][score_name]:.4f}"
            f" [score] {val_score:.4f}"
            f" [loss] {float(np.mean(epoch_losses)):.4f}"
            f" [time] {elapsed:.1f}s"
            f" [it/s] {last_it_s:>3}"
        )
        if epochs_without_improvement >= int(trial_params["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = inference_fn(
            model, dataset_tensors, dataset_meta, trial_params, device, ("train", "val", "test")
        )
    metrics = compute_metrics_for_parts(dataset_meta, predictions, ("train", "val", "test"))
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    report = {
        "model": "mlp",
        "score_name": score_name,
        "score": float(metrics["val"]["score"]),
        "device": str(device),
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch": best_epoch,
        "time_seconds": elapsed,
        "it_s": last_it_s,
        "gpu": capture_gpu_stats(device),
        "metrics": metrics,
        "swap_noise_p": p_swap,
    }
    return report, best_state
