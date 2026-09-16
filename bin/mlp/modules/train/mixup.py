from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
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


def _mixup_batch(
    x_num_batch: Tensor | None,
    x_cat_batch: Tensor | None,
    y_batch: Tensor,
    dataset_meta: Dataset,
    alpha: float = 0.2,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    batch_size = y_batch.shape[0]
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(batch_size, device=y_batch.device)
    x_num_mixed = None
    if x_num_batch is not None:
        x_num_mixed = lam * x_num_batch + (1.0 - lam) * x_num_batch[perm]
    x_cat_mixed = None
    if x_cat_batch is not None:
        cat_mask = (torch.rand(x_cat_batch.shape, device=x_cat_batch.device) < lam).long()
        x_cat_mixed = cat_mask * x_cat_batch + (1 - cat_mask) * x_cat_batch[perm]
    if dataset_meta.is_multiclass:
        y_oh = F.one_hot(y_batch.long(), num_classes=int(dataset_meta.n_classes)).float()
        y_mixed = lam * y_oh + (1.0 - lam) * y_oh[perm]
    else:
        y_mixed = lam * y_batch + (1.0 - lam) * y_batch[perm]
    return x_num_mixed, x_cat_mixed, y_mixed


def train_mixup_v0(
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
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    train_size = dataset_meta.size("train")
    last_it_s = 0
    score_name = dataset_meta.score_name
    mixup_alpha = 0.2
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
            x_num_batch = None if dataset_tensors.x_num is None else dataset_tensors.x_num["train"][batch_idx]
            x_cat_batch = None if dataset_tensors.x_cat is None else dataset_tensors.x_cat["train"][batch_idx]
            y_batch = dataset_tensors.y["train"][batch_idx]
            x_num_mixed, x_cat_mixed, y_mixed = _mixup_batch(
                x_num_batch, x_cat_batch, y_batch, dataset_meta, alpha=mixup_alpha
            )
            preds = model(x_num_mixed, x_cat_mixed)
            loss = loss_fn(preds, y_mixed, dataset_meta)
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
    }
    return report, best_state
