from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.tabm.core import Dataset, TorchDataset
from bin.tabm._util import adjust_gpu_memory_usage


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


def _squeeze_output(preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 3 and preds.shape[-1] == 1:
        return preds.squeeze(-1)
    return preds


def _mean_heads(preds: np.ndarray) -> np.ndarray:
    preds = _squeeze_output(preds)
    if preds.ndim == 2:
        return preds.mean(axis=1)
    return preds


def _to_raw_target_scale(dataset_meta: Dataset, preds: np.ndarray) -> np.ndarray:
    target_inverse_transform = dataset_meta.preprocess_artifacts["target"][
        "inverse_transform"
    ]
    return np.asarray(target_inverse_transform(preds)).reshape(-1).astype(np.float32)


def _predict_with_reducer(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
    reducer: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    for part in parts:
        preds, batch_size = _predict_part(
            model,
            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part],
            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part],
            dataset_tensors.y[part].shape[0],
            dataset_tensors.y[part].device,
            eval_batch_size=batch_size,
        )
        predictions[part] = _to_raw_target_scale(dataset_meta, reducer(preds))
    trial_params["eval_batch_size"] = batch_size
    return predictions


def inference_v0(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _predict_with_reducer(
        model, dataset_tensors, dataset_meta, trial_params, parts, _mean_heads
    )


__all__ = [
    "_mean_heads",
    "_predict_part",
    "_predict_with_reducer",
    "_squeeze_output",
    "_to_raw_target_scale",
    "inference_v0",
]
