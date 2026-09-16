from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset


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
    for part in parts:
        outputs: list[Tensor] = []
        indices = torch.arange(dataset_tensors.y[part].shape[0], device=dataset_tensors.y[part].device)
        for batch_idx in indices.split(batch_size):
            outputs.append(
                model(
                    None if dataset_tensors.x_num is None else dataset_tensors.x_num[part][batch_idx],
                    None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part][batch_idx],
                )
            )
        logits = torch.cat(outputs).detach().cpu().numpy()
        predictions[part] = logits.astype(np.float32)
    return predictions


__all__ = ["inference_v0"]
