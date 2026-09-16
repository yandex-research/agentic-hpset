from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bin.tabm.core import Dataset, TorchDataset

from .inference import _predict_with_reducer, _squeeze_output


def _median_heads(preds: np.ndarray) -> np.ndarray:
    preds = _squeeze_output(preds)
    if preds.ndim == 2:
        return np.median(preds, axis=1)
    return preds


def inference_v1(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _predict_with_reducer(
        model, dataset_tensors, dataset_meta, trial_params, parts, _median_heads
    )


__all__ = ["inference_v1"]
