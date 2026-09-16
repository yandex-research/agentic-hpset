from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bin.tabm.core import Dataset, TorchDataset

from .inference import _predict_with_reducer, _squeeze_output


def _geometric_heads(preds: np.ndarray) -> np.ndarray:
    preds = _squeeze_output(preds)
    if preds.ndim == 2:
        mean_pred = preds.mean(axis=1)
        sign = np.sign(mean_pred)
        sign = np.where(sign == 0, 1.0, sign)
        magnitude = np.exp(np.mean(np.log(np.abs(preds) + 1e-8), axis=1))
        return sign * magnitude
    return preds


def inference_v3(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _predict_with_reducer(
        model, dataset_tensors, dataset_meta, trial_params, parts, _geometric_heads
    )


__all__ = ["inference_v3"]
