from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bin.tabm.core import Dataset, TorchDataset

from .inference import _predict_with_reducer, _squeeze_output


def _variance_weighted_heads(preds: np.ndarray) -> np.ndarray:
    preds = _squeeze_output(preds)
    if preds.ndim == 2:
        mean_pred = preds.mean(axis=1)
        median_pred = np.median(preds, axis=1)
        var = preds.var(axis=1)
        scale = np.median(var)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.mean(var)) + 1e-8
        alpha = np.exp(-var / (scale + 1e-12))
        return alpha * mean_pred + (1.0 - alpha) * median_pred
    return preds


def inference_v6(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _predict_with_reducer(
        model,
        dataset_tensors,
        dataset_meta,
        trial_params,
        parts,
        _variance_weighted_heads,
    )


__all__ = ["inference_v6"]
