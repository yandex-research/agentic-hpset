from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.tabm.core import Dataset, TorchDataset, seed_everything
from bin.tabm._util import adjust_gpu_memory_usage

from .inference import _mean_heads, _to_raw_target_scale


@adjust_gpu_memory_usage("eval_batch_size")
def _predict_part_tta(
    model: torch.nn.Module,
    x_num_part: Tensor | None,
    x_cat_part: Tensor | None,
    n_rows: int,
    reference_device: torch.device,
    *,
    eval_batch_size: int,
    tta_samples: int,
    tta_sigma: float,
) -> np.ndarray:
    outputs: list[Tensor] = []
    indices = torch.arange(n_rows, device=reference_device)
    for batch_idx in indices.split(eval_batch_size):
        x_num_b = None if x_num_part is None else x_num_part[batch_idx]
        x_cat_b = None if x_cat_part is None else x_cat_part[batch_idx]
        tta_outputs: list[Tensor] = []
        for sample_idx in range(max(1, tta_samples)):
            if x_num_b is not None and sample_idx > 0:
                x_num_sample = x_num_b + torch.randn_like(x_num_b) * tta_sigma
            else:
                x_num_sample = x_num_b
            tta_outputs.append(model(x_num_sample, x_cat_b))
        outputs.append(torch.stack(tta_outputs, dim=0).mean(dim=0))
    return torch.cat(outputs).detach().cpu().numpy()


def inference_v4(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    tta_samples = int(trial_params.get("tta_samples", 5))
    tta_sigma = float(trial_params.get("tta_sigma", 0.05))
    seed_everything(int(trial_params.get("seed", 0)) + 7919)
    for part in parts:
        preds, batch_size = _predict_part_tta(
            model,
            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part],
            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part],
            dataset_tensors.y[part].shape[0],
            dataset_tensors.y[part].device,
            eval_batch_size=batch_size,
            tta_samples=tta_samples,
            tta_sigma=tta_sigma,
        )
        predictions[part] = _to_raw_target_scale(dataset_meta, _mean_heads(preds))
    trial_params["eval_batch_size"] = batch_size
    return predictions


__all__ = ["inference_v4"]
