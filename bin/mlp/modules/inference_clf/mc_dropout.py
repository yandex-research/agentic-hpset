"""MC-dropout snapshot averaging.

Runs `K=8` forward passes with Dropout layers kept in train mode (other
modules — BatchNorm, etc. — stay in eval mode) and averages the resulting
logits. Gal & Ghahramani 2016 show this approximates Bayesian model
averaging. If the model has no active Dropout (e.g. dropout=0 trial), the
function collapses gracefully back to a single deterministic pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset


_K = 8


def _enable_only_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _has_active_dropout(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, nn.Dropout) and m.p > 0.0:
            return True
    return False


def inference_v3(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params["eval_batch_size"])
    n_passes = _K if _has_active_dropout(model) else 1
    saved_training = model.training
    if n_passes > 1:
        _enable_only_dropout(model)
    else:
        model.eval()
    try:
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
                for _ in range(n_passes):
                    logits = model(x_num_b, x_cat_b)
                    agg = logits if agg is None else (agg + logits)
                assert agg is not None
                outputs.append(agg / float(n_passes))
            predictions[part] = torch.cat(outputs).detach().cpu().numpy().astype(np.float32)
    finally:
        if saved_training:
            model.train()
        else:
            model.eval()
    return predictions


__all__ = ["inference_v3"]
