from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from bin.tabm.core import (
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


def _copy_state(model: nn.Module) -> dict[str, Tensor]:
    return copy.deepcopy(model.state_dict())


def _iter_batches(
    train_size: int,
    ensemble_size: int,
    batch_size: int,
    device: torch.device,
) -> Iterable[Tensor]:
    if ensemble_size > 1:
        permutation = torch.rand((train_size, ensemble_size), device=device).argsort(dim=0)
        return permutation.split(batch_size, dim=0)
    return torch.randperm(train_size, device=device).split(batch_size)


def _batch_tensors(
    dataset_tensors: TorchDataset, batch_idx: Tensor
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    x_num = (
        None
        if dataset_tensors.x_num is None
        else dataset_tensors.x_num["train"][batch_idx]
    )
    x_cat = (
        None
        if dataset_tensors.x_cat is None
        else dataset_tensors.x_cat["train"][batch_idx]
    )
    y = dataset_tensors.y["train"][batch_idx]
    return x_num, x_cat, y


def _evaluate(
    model: nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    inference_fn: InferenceFn,
    device: torch.device,
    parts: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    predictions = inference_fn(
        model, dataset_tensors, dataset_meta, trial_params, device, parts
    )
    if not all(np.isfinite(predictions[part]).all() for part in predictions):
        raise RuntimeError("Encountered non-finite TabM predictions.")
    return compute_metrics_for_parts(dataset_meta, predictions, parts), predictions


def _print_epoch(
    *,
    improved: bool,
    model_name: str,
    epoch: int,
    score_name: str,
    metrics: dict[str, dict[str, float]],
    val_score: float,
    loss: float,
    elapsed: float,
    it_s: int,
    suffix: str = "",
) -> None:
    # `test` is not evaluated during the epoch loop (only val drives early
    # stopping), so fall back to a placeholder when it is absent.
    test_metric = metrics.get("test", {}).get(score_name)
    test_str = f"{test_metric:.4f}" if test_metric is not None else "  -   "
    print(
        f"{'*' if improved else ' '} [{model_name} epoch] {epoch:<3}"
        f" [val {score_name}] {metrics['val'][score_name]:.4f}"
        f" [test {score_name}] {test_str}"
        f" [score] {val_score:.4f}"
        f" [loss] {loss:.4f}"
        f" [time] {elapsed:.1f}s"
        f" [it/s] {it_s:>3}"
        f"{suffix}"
    )


def _report(
    *,
    model: nn.Module,
    model_name: str,
    dataset_meta: Dataset,
    device: torch.device,
    best_epoch: int,
    elapsed: float,
    it_s: int,
    metrics: dict[str, dict[str, float]],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "model": model_name,
        "score_name": dataset_meta.score_name,
        "score": float(metrics["val"]["score"]),
        "device": str(device),
        "n_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "n_ensemble_members": int(getattr(model, "k", 1)),
        "best_epoch": best_epoch,
        "time_seconds": elapsed,
        "it_s": it_s,
        "gpu": capture_gpu_stats(device),
        "metrics": metrics,
    }
    if extras:
        report.update(extras)
    return report


def train_v0(
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
            loss = loss_fn(model(x_num, x_cat), targets, dataset_meta)
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


__all__ = [
    "InferenceFn",
    "LossFn",
    "OptimizerBuilder",
    "_batch_tensors",
    "_copy_state",
    "_evaluate",
    "_iter_batches",
    "_print_epoch",
    "_report",
    "train_v0",
]
