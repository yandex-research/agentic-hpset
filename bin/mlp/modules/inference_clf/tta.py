"""Test-time augmentation via input noise.

For each row, runs `K=8` forward passes: one unmodified and `K-1` with small
Gaussian noise (`1e-2`) added to the numeric features (categoricals are
untouched so categories don't shift). The mean logit across passes is
returned. This is a single-model variance-reduction trick that smooths
decision boundaries at the cost of K times more inference compute.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset


_K = 8
_NOISE_SCALE = 1e-2


def inference_v1(
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
        indices = torch.arange(
            dataset_tensors.y[part].shape[0], device=dataset_tensors.y[part].device
        )
        for batch_idx in indices.split(batch_size):
            x_num_b = (
                None if dataset_tensors.x_num is None else dataset_tensors.x_num[part][batch_idx]
            )
            x_cat_b = (
                None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part][batch_idx]
            )
            agg: Tensor | None = None
            for k in range(_K):
                if x_num_b is None or k == 0:
                    x_num_aug = x_num_b
                else:
                    x_num_aug = x_num_b + _NOISE_SCALE * torch.randn_like(x_num_b)
                logits = model(x_num_aug, x_cat_b)
                agg = logits if agg is None else (agg + logits)
            assert agg is not None
            outputs.append(agg / float(_K))
        predictions[part] = torch.cat(outputs).detach().cpu().numpy().astype(np.float32)
    return predictions


__all__ = ["inference_v1"]
