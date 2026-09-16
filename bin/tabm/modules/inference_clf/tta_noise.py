from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.tabm.core import Dataset, TorchDataset
from bin.tabm._util import adjust_gpu_memory_usage

from .inference import _aggregate_logits


def _logits_to_probs(logits: np.ndarray, dataset_meta: Dataset) -> np.ndarray:
    if dataset_meta.is_binclass:
        # Overflow on large-negative logits yields prob 0 (correct); silence the
        # benign RuntimeWarning without changing the result.
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def _probs_to_logits(probs: np.ndarray, dataset_meta: Dataset) -> np.ndarray:
    if dataset_meta.is_binclass:
        probs = np.clip(probs.reshape(-1), 1e-7, 1.0 - 1e-7)
        return np.log(probs / (1.0 - probs)).astype(np.float32)
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.log(probs).astype(np.float32)


@adjust_gpu_memory_usage("eval_batch_size")
def _predict_part_tta_logits(
    model: torch.nn.Module,
    x_num_part: torch.Tensor | None,
    x_cat_part: torch.Tensor | None,
    n_rows: int,
    reference_device: torch.device,
    dataset_meta: Dataset,
    *,
    eval_batch_size: int,
    n_aug: int,
    sigma: float,
) -> np.ndarray:
    avg_probs: np.ndarray | None = None
    indices = torch.arange(n_rows, device=reference_device)
    for aug_idx in range(n_aug + 1):
        outputs: list[torch.Tensor] = []
        for batch_idx in indices.split(eval_batch_size):
            x_num = None if x_num_part is None else x_num_part[batch_idx]
            x_cat = None if x_cat_part is None else x_cat_part[batch_idx]
            if aug_idx > 0 and x_num is not None and sigma > 0.0:
                x_num = x_num + torch.randn_like(x_num) * sigma
            outputs.append(model(x_num, x_cat).detach().cpu())
        logits = _aggregate_logits(torch.cat(outputs).numpy(), dataset_meta)
        probs = _logits_to_probs(logits, dataset_meta)
        avg_probs = probs if avg_probs is None else avg_probs + probs
    if avg_probs is None:
        raise RuntimeError("TTA inference did not produce predictions.")
    return _probs_to_logits(avg_probs / float(n_aug + 1), dataset_meta)


def inference_v2(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    n_aug = int(trial_params.get("tta_n_aug", 4))
    sigma = float(trial_params.get("tta_sigma", 0.05))
    for part in parts:
        logits, batch_size = _predict_part_tta_logits(
            model,
            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part],
            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part],
            dataset_tensors.y[part].shape[0],
            dataset_tensors.y[part].device,
            dataset_meta,
            eval_batch_size=batch_size,
            n_aug=n_aug,
            sigma=sigma,
        )
        predictions[part] = logits
    trial_params["eval_batch_size"] = batch_size
    return predictions


__all__ = ["_predict_part_tta_logits", "inference_v2"]
