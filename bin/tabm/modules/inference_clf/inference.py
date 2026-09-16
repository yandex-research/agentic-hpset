from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from bin.tabm.core import Dataset, TorchDataset
from bin.tabm._util import adjust_gpu_memory_usage


def _aggregate_logits(logits: np.ndarray, dataset_meta: Dataset) -> np.ndarray:
    if dataset_meta.is_binclass:
        if logits.ndim == 2:
            # Large-negative logits overflow exp() to +inf -> prob 0 (correct);
            # silence the benign RuntimeWarning without changing the result.
            with np.errstate(over="ignore"):
                probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs.mean(axis=1), 1e-7, 1.0 - 1e-7)
            return np.log(probs / (1.0 - probs)).astype(np.float32)
        return logits.reshape(-1).astype(np.float32)
    if logits.ndim == 3:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        probs = np.clip(probs.mean(axis=1), 1e-7, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return np.log(probs).astype(np.float32)
    return logits.astype(np.float32)


@adjust_gpu_memory_usage("eval_batch_size")
def _predict_part_raw_logits(
    model: torch.nn.Module,
    x_num_part: Tensor | None,
    x_cat_part: Tensor | None,
    n_rows: int,
    reference_device: torch.device,
    *,
    eval_batch_size: int,
) -> np.ndarray:
    outputs: list[Tensor] = []
    indices = torch.arange(n_rows, device=reference_device)
    for batch_idx in indices.split(eval_batch_size):
        outputs.append(
            model(
                None if x_num_part is None else x_num_part[batch_idx],
                None if x_cat_part is None else x_cat_part[batch_idx],
            )
            .detach()
            .cpu()
        )
    return torch.cat(outputs).numpy()


def _predict_raw_logits(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    logits_by_part: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    for part in parts:
        logits, batch_size = _predict_part_raw_logits(
            model,
            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part],
            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part],
            dataset_tensors.y[part].shape[0],
            dataset_tensors.y[part].device,
            eval_batch_size=batch_size,
        )
        logits_by_part[part] = logits
    trial_params["eval_batch_size"] = batch_size
    return logits_by_part


def _predict_aggregated_logits(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    raw_logits = _predict_raw_logits(model, dataset_tensors, trial_params, parts)
    return {
        part: _aggregate_logits(logits, dataset_meta)
        for part, logits in raw_logits.items()
    }


def _fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    dataset_meta: Dataset,
    max_iter: int = 100,
) -> float:
    if logits.size == 0:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(
        y_true,
        dtype=torch.float32 if dataset_meta.is_binclass else torch.long,
    )

    def nll_at(temperature: float) -> float:
        scaled = logits_t / max(temperature, 1e-6)
        if dataset_meta.is_binclass:
            return float(F.binary_cross_entropy_with_logits(scaled, y_t).item())
        return float(F.cross_entropy(scaled, y_t).item())

    baseline_nll = nll_at(1.0)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        scaled = logits_t / torch.exp(log_temperature)
        if dataset_meta.is_binclass:
            loss = F.binary_cross_entropy_with_logits(scaled, y_t)
        else:
            loss = F.cross_entropy(scaled, y_t)
        loss.backward()
        return loss

    try:
        with torch.enable_grad():
            optimizer.step(closure)
        temperature = float(torch.exp(log_temperature.detach()).item())
    except Exception:
        return 1.0
    if (
        not np.isfinite(temperature)
        or temperature <= 0.0
        or temperature < 0.01
        or temperature > 100.0
    ):
        return 1.0
    if nll_at(temperature) >= baseline_nll - 1e-6:
        return 1.0
    return temperature


def inference_v0(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _predict_aggregated_logits(
        model, dataset_tensors, dataset_meta, trial_params, parts
    )


__all__ = [
    "_aggregate_logits",
    "_fit_temperature",
    "_predict_aggregated_logits",
    "_predict_part_raw_logits",
    "_predict_raw_logits",
    "inference_v0",
]
