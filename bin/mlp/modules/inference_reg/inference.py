from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset
from bin.mlp._util import adjust_gpu_memory_usage

from .utils import mean_regression_output


@adjust_gpu_memory_usage("eval_batch_size")
def _predict_part(
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
        )
    return torch.cat(outputs).detach().cpu().numpy()


def inference_v0(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    target_inverse_transform = dataset_meta.preprocess_artifacts["target"]["inverse_transform"]
    for part in parts:
        preds, batch_size = _predict_part(
            model,
            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part],
            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part],
            dataset_tensors.y[part].shape[0],
            dataset_tensors.y[part].device,
            eval_batch_size=batch_size,
        )
        preds = mean_regression_output(preds)
        predictions[part] = target_inverse_transform(preds).astype(np.float32)
    trial_params["eval_batch_size"] = batch_size
    return predictions


__all__ = ["inference_v0"]
