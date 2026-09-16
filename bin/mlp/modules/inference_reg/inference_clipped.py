from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset

from .utils import mean_regression_output


def inference_v2(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Predictions clipped to the training target range."""
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    target_inverse_transform = dataset_meta.preprocess_artifacts["target"]["inverse_transform"]
    y_train_raw = dataset_meta.y_raw["train"]
    clip_min = float(y_train_raw.min())
    clip_max = float(y_train_raw.max())
    for part in parts:
        outputs: list[Tensor] = []
        indices = torch.arange(
            dataset_tensors.y[part].shape[0], device=dataset_tensors.y[part].device
        )
        for batch_idx in indices.split(batch_size):
            outputs.append(
                model(
                    None if dataset_tensors.x_num is None else dataset_tensors.x_num[part][batch_idx],
                    None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part][batch_idx],
                )
            )
        preds = torch.cat(outputs).detach().cpu().numpy()
        preds = mean_regression_output(preds)
        preds = target_inverse_transform(preds).astype(np.float32)
        preds = np.clip(preds, clip_min, clip_max)
        predictions[part] = preds
    return predictions


__all__ = ["inference_v2"]
